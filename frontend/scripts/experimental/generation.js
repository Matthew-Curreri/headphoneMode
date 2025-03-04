// LLaMA model implementation in vanilla JavaScript using WebGPU for acceleration.
// This implementation mirrors a Python version, including tokenization, model architecture,
// and output formatting (chat and text completion). It avoids external libraries and uses 
// WebGPU for matrix operations and parallelism. Logging and debug outputs can be enabled 
// to trace internal states similar to the Python version.

// **Tokenizer Implementation**
// The LLaMA tokenizer is implemented from scratch. We assume a vocabulary mapping tokens to IDs 
// is available (could be loaded from a JSON). Here we build a trie for efficient encoding.
class Tokenizer {
    constructor(vocab) {
        this.token_to_id = vocab;               // Map from token (string) to integer ID
        this.id_to_token = {};                  // Reverse map for decoding
        for (const [tok, id] of Object.entries(vocab)) {
            this.id_to_token[id] = tok;
        }
        // Build trie for tokenization (to find longest matching token prefix)
        this.trie = {};
        for (const [tok, id] of Object.entries(vocab)) {
            let node = this.trie;
            for (const ch of tok) {
                if (!node[ch]) node[ch] = {};
                node = node[ch];
            }
            node.end = id;  // mark end of token with its ID
        }
    }

    // Encode a text string into an array of token IDs
    encode(text) {
        const tokens = [];
        let i = 0;
        const n = text.length;
        while (i < n) {
            let node = this.trie;
            let lastMatchId = null;
            let lastMatchLength = 0;
            let j = i;
            // Walk the trie to find the longest token that is a prefix of text[i:]
            while (j < n && node[text[j]]) {
                node = node[text[j]];
                j++;
                if (node.end !== undefined) {  // found a token
                    lastMatchId = node.end;
                    lastMatchLength = j - i;
                }
            }
            if (lastMatchId === null) {
                // No token matched (should not happen if vocab covers all characters). 
                // As fallback, use unknown token ID (assumed 0 if present).
                lastMatchId = this.token_to_id["<unk>"] || 0;
                lastMatchLength = 1;
            }
            tokens.push(lastMatchId);
            i += lastMatchLength;
        }
        return tokens;
    }

    // Decode an array of token IDs back into the text string
    decode(tokenIds) {
        let text = "";
        for (const id of tokenIds) {
            const tok = this.id_to_token[id];
            if (tok === undefined) continue;  // skip unknown id
            text += tok;
        }
        return text;
    }
}

// **Top-p (nucleus) Sampling Implementation**
// Given a logits array (raw probabilities before softmax) for the next token, 
// select a token ID using nucleus sampling&#8203;:contentReference[oaicite:0]{index=0}. This limits selection 
// to the smallest set of tokens whose cumulative probability exceeds the threshold p.
function sampleTopP(logits, top_p=0.9, temperature=1.0) {
    // Apply temperature scaling: divide logits by temperature
    const tempAdjusted = logits.map(x => x / temperature);
    // Compute softmax probabilities in a stable way
    // Find max logit for numerical stability
    const maxLogit = Math.max(...tempAdjusted);
    const expScores = tempAdjusted.map(x => Math.exp(x - maxLogit));
    const sumExp = expScores.reduce((a, b) => a + b, 0);
    let probs = expScores.map(x => x / sumExp);
    // Sort token indices by probability descending
    const sortedIndices = probs.map((p,i)=>[p,i]).sort((a,b) => b[0]-a[0]);
    // Determine cutoff where cumulative probability > top_p
    let cumProb = 0;
    let cutoffIndex = 0;
    for (; cutoffIndex < sortedIndices.length; cutoffIndex++) {
        cumProb += sortedIndices[cutoffIndex][0];
        if (cumProb >= top_p) break;
    }
    const filtered = sortedIndices.slice(0, cutoffIndex+1);
    // Re-normalize probabilities of the remaining tokens
    const filteredProbSum = filtered.reduce((a, [p, _]) => a + p, 0);
    for (let k = 0; k < filtered.length; k++) {
        filtered[k][0] = filtered[k][0] / filteredProbSum;
    }
    // Randomly pick one token from the filtered set
    const rand = Math.random();
    let accum = 0;
    for (const [p, idx] of filtered) {
        accum += p;
        if (rand < accum) {
            return idx;
        }
    }
    // Fallback (shouldn't normally happen if probabilities sum to 1)
    return filtered[filtered.length - 1][1];
}

// **LLaMA Model Implementation**
class LlamaModel {
    constructor(config, device) {
        this.config = config;
        this.device = device;
        // Model hyperparameters
        this.n_layers = config.num_layers || config.n_layers;
        this.n_heads = config.num_heads || config.n_heads;
        this.d_model = config.hidden_size || config.d_model;
        this.vocab_size = config.vocab_size;
        this.head_dim = Math.floor(this.d_model / this.n_heads);
        this.ffn_dim = config.ffn_hidden_size || config.ffn_dim || (this.d_model * 4);  // feed-forward intermediate dim
        this.epsilon = config.norm_eps || 1e-6;   // RMSNorm epsilon for numerical stability
        this.use_gpu = !!device;  // use GPU if device available
        this.debug = config.debug || false;
        // Containers for model weights (each weight as Float32Array or GPUBuffer)
        this.weights = {};
        // Cache for key and value tensors for each transformer layer (for fast generation)
        // Each entry: { K: Array of Float32Array (each of length d_model) for past keys, V: similarly for past values }
        this.cache = Array.from({length: this.n_layers}, () => ({K: [], V: []}));
        // Precompute rotary embedding cosines and sines for positions up to max_seq_len
        this.max_seq_len = config.max_seq_len || 2048;
        const headDim = this.head_dim;
        const baseFreq = 10000;  // RoPE base
        // Compute inverse frequency for each pair dimension index
        const invFreq = new Float32Array(headDim / 2);
        for (let i = 0; i < invFreq.length; i++) {
            invFreq[i] = 1.0 / Math.pow(baseFreq, i / invFreq.length);
        }
        this.rotary_cos = new Float32Array(this.max_seq_len * invFreq.length);
        this.rotary_sin = new Float32Array(this.max_seq_len * invFreq.length);
        for (let pos = 0; pos < this.max_seq_len; pos++) {
            for (let i = 0; i < invFreq.length; i++) {
                const angle = pos * invFreq[i];
                this.rotary_cos[pos * invFreq.length + i] = Math.cos(angle);
                this.rotary_sin[pos * invFreq.length + i] = Math.sin(angle);
            }
        }
        // Initialize WebGPU compute pipeline for matrix multiplication
        this._createMatMulPipeline();
    }

    // **Model weight loading from JSON format**
    // Expects a JSON with model config and weights (could be all weights in arrays, or a structure referencing binary).
    static async loadFromJSON(jsonUrl, weightUrl=null) {
        const response = await fetch(jsonUrl);
        const modelData = await response.json();
        // If weights are stored separately in binary file(s), handle accordingly
        let binBuffer = null;
        if (weightUrl) {
            binBuffer = await fetch(weightUrl).then(res => res.arrayBuffer());
        } else if (modelData.model_bin) {
            // If JSON references a binary file
            binBuffer = await fetch(modelData.model_bin).then(res => res.arrayBuffer());
        }
        // Request WebGPU adapter and device
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("WebGPU not supported or available in this environment.");
        const device = await adapter.requestDevice();
        const model = new LlamaModel(modelData, device);
        if (binBuffer) {
            // If we have a binary buffer and an index of weight offsets
            if (!modelData.weight_map) {
                throw new Error("No weight_map found for binary weights.");
            }
            model._loadWeightsFromBinary(modelData.weight_map, binBuffer);
        } else if (modelData.weights) {
            // Weights provided directly in JSON (as nested lists or arrays)
            model._loadWeightsFromJSON(modelData.weights);
        } else {
            throw new Error("No weight data found in JSON.");
        }
        return model;
    }

    // **Model weight loading from binary format**
    // This expects a JSON index (configUrl) describing model hyperparams and weight offsets, and a binary file for weights.
    static async loadFromBinary(configUrl, weightsUrl) {
        const [configData, weightsBuffer] = await Promise.all([
            fetch(configUrl).then(res => res.json()),
            fetch(weightsUrl).then(res => res.arrayBuffer())
        ]);
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("WebGPU not supported or available.");
        const device = await adapter.requestDevice();
        const model = new LlamaModel(configData, device);
        if (!configData.weight_map) {
            throw new Error("Config missing weight_map for binary weights.");
        }
        model._loadWeightsFromBinary(configData.weight_map, weightsBuffer);
        return model;
    }

    // Update loadFromBinary to accept binary data directly
    static async loadFromBinary(configUrl, binaryData) {
        const configData = await fetch(configUrl).then(res => res.json());
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("WebGPU not supported or available.");
        const device = await adapter.requestDevice();
        const model = new LlamaModel(configData, device);
        if (!configData.weight_map) {
            throw new Error("Config missing weight_map for binary weights.");
        }
        model._loadWeightsFromBinary(configData.weight_map, binaryData);
        return model;
    }

    // Internal helper: load weights from a JSON structure (object of name: values).
    _loadWeightsFromJSON(weightsObj) {
        for (const [name, value] of Object.entries(weightsObj)) {
            // Convert array (or nested arrays) to Float32Array
            let arr = value;
            if (!(arr instanceof Float32Array)) {
                arr = new Float32Array(value);
            }
            // If GPU is available, create GPUBuffer for the weight
            if (this.use_gpu) {
                this.weights[name] = this._uploadAndCreateBuffer(arr, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
            } else {
                this.weights[name] = arr;
            }
            if (this.debug) {
                console.log(`Loaded weight ${name} [length=${arr.length}]`);
            }
        }
    }

    // Internal helper: load weights from a binary buffer using an index map of offsets and lengths.
    _loadWeightsFromBinary(weightMap, buffer) {
        const dataView = new DataView(buffer);
        for (const [name, spec] of Object.entries(weightMap)) {
            const offset = spec.offset;    // offset index in float (not bytes)
            const length = spec.length;    // number of float32 elements
            const byteOffset = offset * 4;
            const byteLength = length * 4;
            const floatArray = new Float32Array(buffer, byteOffset, length);
            // Copy the slice to avoid referencing the whole buffer (if needed)
            const weights = new Float32Array(floatArray); 
            if (this.use_gpu) {
                this.weights[name] = this._uploadAndCreateBuffer(weights, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
            } else {
                this.weights[name] = weights;
            }
            if (this.debug) {
                console.log(`Loaded weight ${name} [length=${length}]`);
            }
        }
    }

    // **WebGPU Setup: Matrix Multiply Compute Pipeline**
    // Create a GPU compute pipeline for matrix multiplication (for general MxK * KxN). We use WGSL shader code.
    _createMatMulPipeline() {
        if (!this.device) return;
        // Define WGSL shader for matrix multiplication: it multiplies matrices A (MxK) and B (KxN) into C (MxN).
        const shaderCode = `
            @group(0) @binding(0) var<storage, read> A: array<f32>;
            @group(0) @binding(1) var<storage, read> B: array<f32>;
            @group(0) @binding(2) var<storage, read_write> C: array<f32>;
            @group(0) @binding(3) var<uniform> dims: vec3<u32>; // (M, N, K)
            @compute @workgroup_size(8, 8)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let M = dims.x;
                let N = dims.y;
                let K = dims.z;
                let row = global_id.x;
                let col = global_id.y;
                // Bounds check for matrix dimensions&#8203;:contentReference[oaicite:1]{index=1}
                if (row >= M || col >= N) {
                    return;
                }
                var sum = 0.0;
                for (var i: u32 = 0u; i < K; i = i + 1u) {
                    // Compute dot product for element (row, col)&#8203;:contentReference[oaicite:2]{index=2}
                    sum = sum + A[row * K + i] * B[i * N + col];
                }
                C[row * N + col] = sum;
            }
        `;
        const shaderModule = this.device.createShaderModule({ code: shaderCode });
        // Create bind group layout: 0->A, 1->B, 2->C, 3->dims
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
            ]
        });
        // Create the compute pipeline using the bind group layout and shader
        this.matMulPipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module: shaderModule, entryPoint: "main" }
        });
    }

    // Helper to create a GPUBuffer from a Float32Array and upload data
    _uploadAndCreateBuffer(dataArray, usage) {
        const buffer = this.device.createBuffer({
            size: dataArray.byteLength,
            usage: usage || (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC),
            mappedAtCreation: true
        });
        const arrayBuffer = buffer.getMappedRange();
        new Float32Array(arrayBuffer).set(dataArray);
        buffer.unmap();
        return buffer;
    }

    // Helper to run a GPU matmul: multiplies A (M x K) with B (K x N), returns result as Float32Array of shape (M x N).
    _runMatMul(A, B, M, N, K) {
        if (!this.device || !this.matMulPipeline) {
            throw new Error("WebGPU device not initialized for matMul.");
        }
        // Ensure A and B are GPUBuffer (upload data if they are Float32Array)
        let aBuffer = A;
        if (!(A instanceof GPUBuffer)) {
            aBuffer = this._uploadAndCreateBuffer(A, GPUBufferUsage.STORAGE);
        }
        let bBuffer = B;
        if (!(B instanceof GPUBuffer)) {
            bBuffer = this._uploadAndCreateBuffer(B, GPUBufferUsage.STORAGE);
        }
        // Create output buffer C for result (M*N floats)
        const resultSize = M * N;
        const cBuffer = this.device.createBuffer({
            size: resultSize * 4, // float32 bytes
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        // Create uniform buffer for dims (M, N, K)
        const dimsBuffer = this.device.createBuffer({
            size: 3 * 4, // 3 x 32-bit
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true
        });
        // Set uniform data
        new Uint32Array(dimsBuffer.getMappedRange()).set([M, N, K]);
        dimsBuffer.unmap();
        // Create bind group for this computation
        const bindGroup = this.device.createBindGroup({
            layout: this.matMulPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: aBuffer } },
                { binding: 1, resource: { buffer: bBuffer } },
                { binding: 2, resource: { buffer: cBuffer } },
                { binding: 3, resource: { buffer: dimsBuffer } }
            ]
        });
        // Dispatch compute work
        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(this.matMulPipeline);
        passEncoder.setBindGroup(0, bindGroup);
        // Use a workgroup grid covering MxN threads (8x8 group size, so divide ceiling by 8)
        const workgroupsX = Math.ceil(M / 8);
        const workgroupsY = Math.ceil(N / 8);
        passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);
        passEncoder.end();
        this.device.queue.submit([commandEncoder.finish()]);
        // Read back the result from GPU
        // We create a read buffer and copy the result into it for mapping
        const readBuffer = this.device.createBuffer({
            size: resultSize * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const cmdEncoder2 = this.device.createCommandEncoder();
        cmdEncoder2.copyBufferToBuffer(cBuffer, 0, readBuffer, 0, resultSize * 4);
        this.device.queue.submit([cmdEncoder2.finish()]);
        // Now map the readBuffer to get the results
        return readBuffer.mapAsync(GPUMapMode.READ).then(() => {
            const copyArrayBuffer = readBuffer.getMappedRange();
            const resultArray = new Float32Array(copyArrayBuffer.slice(0));  // copy the data out
            readBuffer.unmap();
            // Clean up temporary GPU buffers (A and B if we created new ones)
            // (In a real implementation, reuse buffers to avoid overhead).
            if (!(A instanceof GPUBuffer)) aBuffer.destroy();
            if (!(B instanceof GPUBuffer)) bBuffer.destroy();
            cBuffer.destroy();
            dimsBuffer.destroy();
            readBuffer.destroy();
            return resultArray;
        });
    }

    // RMSNorm: normalize a vector using root-mean-square norm, then scale by weight.
    _rmsNorm(vec, weight) {
        // vec and weight are Float32Array (length = d_model).
        const n = vec.length;
        let sumSquares = 0.0;
        for (let i = 0; i < n; i++) {
            const val = vec[i];
            sumSquares += val * val;
        }
        const meanSq = sumSquares / n;
        const invNorm = 1.0 / Math.sqrt(meanSq + this.epsilon);
        const out = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            out[i] = vec[i] * invNorm * weight[i];
        }
        return out;
    }

    // Clear the KV cache (e.g., before processing a new prompt).
    clearCache() {
        this.cache = Array.from({length: this.n_layers}, () => ({K: [], V: []}));
    }

    // **Forward pass (inference) for the model.**
    // If use_cache is true, we assume `tokens` is an array of 1 token ID (the latest token to process),
    // and that `this.cache` contains the accumulated keys/values for previous tokens.
    // If use_cache is false, we process the entire sequence of tokens from scratch (useful for first-time prompt).
    // Returns the logits (unnormalized probabilities) for the next token.
    async forward(tokens, use_cache=false) {
        // If GPU is not used for embedding lookup or small ops, proceed with CPU for those parts.
        // 1. Get embeddings for input tokens
        const seqLength = tokens.length;
        const batchSize = 1;  // this implementation assumes a single sequence (batch=1). 
        // Prepare hidden state array of shape (seqLength, d_model)
        // If use_cache and tokens.length == 1, we will only compute for this single token (position = current seq length in cache).
        let hiddenStates;
        if (use_cache && seqLength === 1) {
            hiddenStates = new Float32Array(this.d_model);
            const tokenId = tokens[0];
            // Embedding lookup for the single token
            // Weight name might be 'tok_embeddings' or similar:
            const embWeight = this.weights['tok_embeddings'] || this.weights['token_embedding.weight'] || this.weights['token_embeddings'];
            if (!embWeight) throw new Error("Token embedding weight not found");
            if (embWeight instanceof GPUBuffer) {
                // If embedding matrix is on GPU, we perform embedding lookup manually:
                // Since we cannot easily index a GPUBuffer without mapping, do on CPU by reading the required row.
                // (An optimization could be a specialized GPU kernel to gather embeddings.)
                // Map embedding buffer (if huge, this is expensive; ideally would have the embedding in CPU memory as well).
                // For simplicity, assume we have a CPU copy in config (not shown here).
                throw new Error("Embedding lookup on GPU not implemented in this snippet");
            } else {
                const vocabSize = this.vocab_size;
                const dim = this.d_model;
                // The embedding weight is flattened [vocab_size * d_model]
                // Find the row for tokenId
                const offset = tokenId * dim;
                hiddenStates.set(embWeight.subarray(offset, offset + dim));
            }
        } else {
            // Processing a full sequence from scratch (or multiple tokens at once)
            hiddenStates = new Float32Array(seqLength * this.d_model);
            const embWeight = this.weights['tok_embeddings'] || this.weights['token_embedding.weight'] || this.weights['token_embeddings'];
            if (!embWeight) throw new Error("Token embedding weight not found");
            if (embWeight instanceof GPUBuffer) {
                // Map the whole embedding (could be large) to CPU to gather multiple tokens
                // (In practice, one might perform this on GPU to avoid huge transfer, but for simplicity we map here.)
                throw new Error("Batch embedding lookup on GPU not implemented in this snippet");
            } else {
                const dim = this.d_model;
                for (let t = 0; t < seqLength; t++) {
                    const tokenId = tokens[t];
                    const offset = tokenId * dim;
                    hiddenStates.set(embWeight.subarray(offset, offset + dim), t * dim);
                }
            }
            // Clear cache because we are doing a fresh pass
            this.clearCache();
        }

        // 2. Iterate through each transformer layer
        // We'll propagate the hidden state for each token position. For efficiency in generation mode, we only compute for the last token using cached keys for attention.
        let offsetLast = (seqLength - 1) * this.d_model;  // offset of last token in hiddenStates array
        for (let layer = 0; layer < this.n_layers; layer++) {
            // Layer normalization (RMSNorm) before self-attention
            const attnNormWeight = this.weights[`layers.${layer}.attention_norm.weight`] || this.weights[`layers.${layer}.attention_norm`];
            if (!attnNormWeight) throw new Error(`Layer ${layer} attention norm weight not found`);
            let attnNorm;
            if (use_cache && seqLength === 1) {
                // For single token, hiddenStates is just one vector
                const hVec = hiddenStates;  // already a Float32Array of length d_model
                attnNorm = this._rmsNorm(hVec, attnNormWeight instanceof GPUBuffer ? 
                                                    // Map GPU buffer to CPU for norm (could optimize by GPU kernel)
                                                    new Float32Array(this.d_model) : attnNormWeight);
            } else {
                // For full sequence: we need to normalize each position vector. 
                // Do it for the last position (for attention calculation for that pos).
                const hVec = hiddenStates.subarray(offsetLast, offsetLast + this.d_model);
                attnNorm = this._rmsNorm(hVec, attnNormWeight instanceof GPUBuffer ? 
                                                    new Float32Array(this.d_model) : attnNormWeight);
                // (In a complete implementation, we would also compute attn outputs for all positions, but for generation we focus on last token.)
            }

            // 3. Self-Attention
            // Compute query, key, value vectors. We use the attentionNorm output as input to linear layers Wq, Wk, Wv.
            const wq = this.weights[`layers.${layer}.attention.wq.weight`] || this.weights[`layers.${layer}.attn_q`];
            const wk = this.weights[`layers.${layer}.attention.wk.weight`] || this.weights[`layers.${layer}.attn_k`];
            const wv = this.weights[`layers.${layer}.attention.wv.weight`] || this.weights[`layers.${layer}.attn_v`];
            if (!wq || !wk || !wv) throw new Error(`Layer ${layer} attention weight missing (Wq/Wk/Wv).`);
            // Multiply attnNorm (1 x d_model) by each weight (d_model x d_model) to get 1 x d_model outputs for Q, K, V.
            // Use GPU if available for these matrix multiplies.
            let Q, K, V;
            if (this.use_gpu) {
                // Run matMul for each (could be parallelized/fused, but we do sequentially here).
                Q = this._runMatMul(attnNorm, wq, 1, this.d_model, this.d_model);
                K = this._runMatMul(attnNorm, wk, 1, this.d_model, this.d_model);
                V = this._runMatMul(attnNorm, wv, 1, this.d_model, this.d_model);
                // The above are Promises because _runMatMul is async (due to GPU readback). Wait for them:
                // (In practice, these could be executed concurrently and synchronized later, but for clarity we do sequential await.)
            } else {
                // On CPU, perform matrix multiplication directly (this is slow for large d_model, so GPU is strongly recommended).
                Q = new Float32Array(this.d_model);
                K = new Float32Array(this.d_model);
                V = new Float32Array(this.d_model);
                const d = this.d_model;
                // Compute Q = attnNorm * Wq (1xd * dxd)
                const wqArr = wq;
                for (let j = 0; j < d; j++) {
                    let sum = 0;
                    for (let i = 0; i < d; i++) {
                        sum += attnNorm[i] * wqArr[i * d + j];
                    }
                    Q[j] = sum;
                }
                // Similarly for K and V
                const wkArr = wk, wvArr = wv;
                for (let j = 0; j < d; j++) {
                    let sumK = 0, sumV = 0;
                    for (let i = 0; i < d; i++) {
                        const val = attnNorm[i];
                        sumK += val * wkArr[i * d + j];
                        sumV += val * wvArr[i * d + j];
                    }
                    K[j] = sumK;
                    V[j] = sumV;
                }
            }
            // If GPU, wait for results
            if (Q instanceof Promise) {
                Q = await Q;
                K = await K;
                V = await V;
            }
            // Convert Q, K, V to dimension [n_heads, head_dim]
            const n_heads = this.n_heads;
            const head_dim = this.head_dim;
            const Q_heads = Array.from({length: n_heads}, (_, h) => Q.subarray(h * head_dim, (h+1) * head_dim));
            const K_new_heads = Array.from({length: n_heads}, (_, h) => K.subarray(h * head_dim, (h+1) * head_dim));
            const V_new_heads = Array.from({length: n_heads}, (_, h) => V.subarray(h * head_dim, (h+1) * head_dim));
            // Apply rotary positional embeddings to Q and K of this token
            // Determine the position index for this token in the sequence:
            let posIndex;
            if (use_cache && seqLength === 1) {
                // If using cache, current seq length is (cache length + 1 for this token)
                const past_len = this.cache[layer].K.length;
                posIndex = past_len;  // 0-based index of new token
            } else {
                posIndex = seqLength - 1;
            }
            for (let h = 0; h < n_heads; h++) {
                const qh = Q_heads[h];
                const kh = K_new_heads[h];
                for (let i = 0; i < head_dim / 2; i++) {
                    const cos = this.rotary_cos[posIndex * (head_dim/2) + i];
                    const sin = this.rotary_sin[posIndex * (head_dim/2) + i];
                    const x0 = qh[2*i], x1 = qh[2*i + 1];
                    // Rotate (x0, x1) by angle (cos, sin) -> (x0*cos - x1*sin, x0*sin + x1*cos)
                    qh[2*i]     = x0 * cos - x1 * sin;
                    qh[2*i + 1] = x0 * sin + x1 * cos;
                    const y0 = kh[2*i], y1 = kh[2*i + 1];
                    kh[2*i]     = y0 * cos - y1 * sin;
                    kh[2*i + 1] = y0 * sin + y1 * cos;
                }
            }
            // Append the new K, V to cache for this layer
            this.cache[layer].K.push(K);  // store the full d_model vector or store per head? (full vector is fine)
            this.cache[layer].V.push(V);
            // Compute self-attention for the new token (query Q against all cached keys K)
            const past_len = this.cache[layer].K.length;  // current sequence length for this layer
            // We'll compute attention output for the last token (index past_len-1)
            const headContext = new Array(n_heads);
            const invSqrt_d = 1.0 / Math.sqrt(head_dim);
            for (let h = 0; h < n_heads; h++) {
                // Get current head's query (qh) and all keys and values for that head from cache
                const qh = Q_heads[h];
                // Prepare to compute attention scores for each past token j
                const scores = new Float32Array(past_len);
                for (let j = 0; j < past_len; j++) {
                    const kj = this.cache[layer].K[j].subarray(h * head_dim, h * head_dim + head_dim);
                    // Dot product qh · kj
                    let score = 0.0;
                    for (let k = 0; k < head_dim; k++) {
                        score += qh[k] * kj[k];
                    }
                    score *= invSqrt_d;
                    // Apply causal mask: if this is for token at index past_len-1, ignore any j > past_len-1 (none in loop),
                    // so no mask needed here explicitly (all j <= past_len-1 by loop definition).
                    // If we were computing for intermediate tokens, we'd ensure not to include future tokens (j > i).
                    scores[j] = score;
                }
                // Softmax over scores[0..past_len-1]
                // Find max for stability
                let maxScore = -Infinity;
                for (let j = 0; j < past_len; j++) {
                    if (scores[j] > maxScore) maxScore = scores[j];
                }
                let sumExp = 0.0;
                for (let j = 0; j < past_len; j++) {
                    scores[j] = Math.exp(scores[j] - maxScore);
                    sumExp += scores[j];
                }
                for (let j = 0; j < past_len; j++) {
                    scores[j] /= sumExp;
                }
                // Weighted sum of values
                const contextVec = new Float32Array(head_dim);
                for (let j = 0; j < past_len; j++) {
                    const vj = this.cache[layer].V[j].subarray(h * head_dim, h * head_dim + head_dim);
                    const w = scores[j];
                    for (let k = 0; k < head_dim; k++) {
                        contextVec[k] += vj[k] * w;
                    }
                }
                headContext[h] = contextVec;
            }
            // Concatenate all head context vectors and project through Wo (d_model x d_model)
            const attnOutput = new Float32Array(this.d_model);
            // Flatten headContext into one vector of length d_model
            const concatenated = new Float32Array(this.d_model);
            for (let h = 0; h < n_heads; h++) {
                concatenated.set(headContext[h], h * head_dim);
            }
            const wo = this.weights[`layers.${layer}.attention.wo.weight`] || this.weights[`layers.${layer}.attn_out`];
            if (!wo) throw new Error(`Layer ${layer} WO weight not found`);
            if (this.use_gpu) {
                // Multiply concatenated (1 x d_model) by Wo (d_model x d_model)
                const result = this._runMatMul(concatenated, wo, 1, this.d_model, this.d_model);
                let attnOutVec = result;
                if (attnOutVec instanceof Promise) attnOutVec = await attnOutVec;
                attnOutput.set(attnOutVec);
            } else {
                // CPU matrix multiply (d_model x d_model)
                const d = this.d_model;
                const woArr = wo;
                for (let j = 0; j < d; j++) {
                    let sum = 0.0;
                    for (let i = 0; i < d; i++) {
                        sum += concatenated[i] * woArr[i * d + j];
                    }
                    attnOutput[j] = sum;
                }
            }
            // Residual connection: add attention output to hidden state
            // For generation mode with single token, hiddenStates is length d_model vector
            // For full sequence, we update only the last token's hidden state (others remain as is from previous steps).
            const hVec = (use_cache && seqLength === 1) ? hiddenStates : hiddenStates.subarray(offsetLast, offsetLast + this.d_model);
            for (let i = 0; i < this.d_model; i++) {
                hVec[i] = hVec[i] + attnOutput[i];
            }

            // 4. Feed-Forward Network (FFN)
            // Apply second RMSNorm
            const ffnNormWeight = this.weights[`layers.${layer}.ffn_norm.weight`] || this.weights[`layers.${layer}.ffn_norm`];
            if (!ffnNormWeight) throw new Error(`Layer ${layer} ffn norm weight not found`);
            const ffnNormVec = this._rmsNorm(hVec, ffnNormWeight instanceof GPUBuffer ? new Float32Array(this.d_model) : ffnNormWeight);
            // Apply feed-forward: LLaMA uses a gated GELU (SwiGLU) with weights w_up, w_gate, and w_down
            const w_up = this.weights[`layers.${layer}.feed_forward.w_up.weight`] || this.weights[`layers.${layer}.ffn_up`];
            const w_down = this.weights[`layers.${layer}.feed_forward.w_down.weight`] || this.weights[`layers.${layer}.ffn_down`];
            const w_gate = this.weights[`layers.${layer}.feed_forward.w_gate.weight`] || this.weights[`layers.${layer}.ffn_gate`];
            if (!w_up || !w_down || !w_gate) throw new Error(`Layer ${layer} FFN weights not found`);
            // Compute X_up = ffnNormVec * w_up  (1 x d_model * d_model x ffn_dim = 1 x ffn_dim)
            // Compute X_gate = ffnNormVec * w_gate (same shape 1 x ffn_dim)
            // Then elementwise: activated = swish(X_up) * X_gate, where swish(x) = x * sigmoid(x)
            // Finally, output = activated * w_down (1 x ffn_dim * ffn_dim x d_model = 1 x d_model)
            let X_up, X_gate, X_down;
            if (this.use_gpu) {
                X_up = this._runMatMul(ffnNormVec, w_up, 1, this.ffn_dim, this.d_model);
                X_gate = this._runMatMul(ffnNormVec, w_gate, 1, this.ffn_dim, this.d_model);
                X_up = await X_up;  X_gate = await X_gate;
            } else {
                X_up = new Float32Array(this.ffn_dim);
                X_gate = new Float32Array(this.ffn_dim);
                const d = this.d_model, fd = this.ffn_dim;
                const w_up_arr = w_up, w_gate_arr = w_gate;
                for (let j = 0; j < fd; j++) {
                    let sumUp = 0.0, sumGate = 0.0;
                    for (let i = 0; i < d; i++) {
                        const val = ffnNormVec[i];
                        sumUp += val * w_up_arr[i * fd + j];
                        sumGate += val * w_gate_arr[i * fd + j];
                    }
                    X_up[j] = sumUp;
                    X_gate[j] = sumGate;
                }
            }
            // Apply activation (Swish) to X_up and multiply by X_gate
            // Swish(x) = x * sigmoid(x)
            const ffnInter = new Float32Array(this.ffn_dim);
            for (let j = 0; j < this.ffn_dim; j++) {
                const u = X_up[j];
                const v = X_gate[j];
                const swishU = u * (1.0 / (1.0 + Math.exp(-u)));  // x * sigmoid(x)
                ffnInter[j] = swishU * v;
            }
            // Now project down: ffnInter (1 x ffn_dim) * w_down (ffn_dim x d_model) -> 1 x d_model
            if (this.use_gpu) {
                X_down = this._runMatMul(ffnInter, w_down, 1, this.d_model, this.ffn_dim);
                X_down = await X_down;
            } else {
                X_down = new Float32Array(this.d_model);
                const d = this.d_model, fd = this.ffn_dim;
                const w_down_arr = w_down;
                for (let j = 0; j < d; j++) {
                    let sum = 0.0;
                    for (let i = 0; i < fd; i++) {
                        sum += ffnInter[i] * w_down_arr[i * d + j];
                    }
                    X_down[j] = sum;
                }
            }
            // Add FFN output to hidden state (residual)
            for (let i = 0; i < this.d_model; i++) {
                hVec[i] = hVec[i] + X_down[i];
            }
            // (At this point, hVec is the output of the layer for the last token. 
            // If fully doing sequence, we'd have outputs for all tokens.)
        } // end of layers loop

        // 5. Final RMSNorm on hidden state
        const finalNormWeight = this.weights['norm.weight'] || this.weights['norm'] || this.weights['output_norm.weight'];
        if (!finalNormWeight) throw new Error("Final RMSNorm weight not found");
        const finalHidden = this._rmsNorm(use_cache && seqLength === 1 ? hiddenStates : hiddenStates.subarray(offsetLast, offsetLast + this.d_model),
                                          finalNormWeight instanceof GPUBuffer ? new Float32Array(this.d_model) : finalNormWeight);
        // 6. Compute logits: finalHidden (1 x d_model) * output weight (d_model x vocab_size) = 1 x vocab_size
        const outputWeight = this.weights['output.weight'] || this.weights['lm_head.weight'] || this.weights['token_embedding.weight'];
        if (!outputWeight) throw new Error("Output weight not found");
        let logits;
        if (this.use_gpu) {
            logits = this._runMatMul(finalHidden, outputWeight, 1, this.vocab_size, this.d_model);
            if (logits instanceof Promise) logits = await logits;
        } else {
            logits = new Float32Array(this.vocab_size);
            const d = this.d_model;
            const V = this.vocab_size;
            const outW = outputWeight;
            for (let j = 0; j < V; j++) {
                let sum = 0.0;
                for (let i = 0; i < d; i++) {
                    sum += finalHidden[i] * outW[i * V + j];
                }
                logits[j] = sum;
            }
        }
        return logits;
    }

    // **Text Generation: complete a text prompt**
    // This uses the model to generate text continuations given a prompt string.
    // It returns an object in a structured format similar to OpenAI's text-completion API.
    async generateTextCompletion(prompt, options={}) {
        const {
            maxNewTokens = 50,
            top_p = 0.9,
            temperature = 1.0,
            stop = null  // optional stop token or string
        } = options;
        if (!this.tokenizer) {
            throw new Error("Tokenizer not set. Please attach a Tokenizer instance to LlamaModel.");
        }
        // Encode prompt to tokens
        const inputIds = this.tokenizer.encode(prompt);
        if (this.debug) {
            console.log(`Prompt tokens: ${inputIds}`);
        }
        // Feed the prompt through the model to prime the cache
        this.clearCache();
        for (let i = 0; i < inputIds.length; i++) {
            // Process each token sequentially using cache (to build up internal state)
            const tok = inputIds[i];
            const logits = await this.forward([tok], true);
            if (this.debug) {
                const tokText = this.tokenizer.decode([tok]);
                console.log(`Input token ${tok} ('${tokText}') processed.`);
            }
        }
        // Now generate new tokens
        const generatedIds = [];
        for (let n = 0; n < maxNewTokens; n++) {
            const logits = await this.forward([generatedIds.length === 0 ? 
                                               inputIds[inputIds.length-1] : generatedIds[generatedIds.length-1]], true);
            // Sample next token using top-p
            const nextTokenId = sampleTopP(logits, top_p, temperature);
            if (this.debug) {
                console.log(`Generated token: ${nextTokenId} ('${this.tokenizer.decode([nextTokenId])}')`);
            }
            // If stop criteria (either EOS token or matching stop string if implemented)
            const eosToken = this.config.eos_token_id || 2;
            if (nextTokenId === eosToken) {
                break;
            }
            generatedIds.push(nextTokenId);
        }
        // Decode the generated tokens to text
        const generatedText = this.tokenizer.decode(generatedIds);
        // Construct structured response object
        return {
            id: "cmpl-" + Math.random().toString(36).substring(2),  // random ID
            object: "text_completion",
            created: Math.floor(Date.now() / 1000),
            model: this.config.model_name || "LlamaModel",
            choices: [
                {
                    index: 0,
                    text: generatedText,
                    finish_reason: "stop"
                }
            ]
        };
    }

    // **Chat Generation: complete a conversation** 
    // Accepts an array of message objects (with roles like 'system', 'user', 'assistant') and generates the assistant's reply.
    // Returns a structured response similar to OpenAI's chat completion API.
    async generateChatCompletion(messages, options={}) {
        let prompt = "Be as helpful and honest as is possible guide the user through the little things in their life.  Delegate tasks to the manager to preform background and scheduled tasks.  This is audio keep your answers short the less words the better if you need a moment to think feel free to use anamaonapia ";
        for (const msg of messages) {
            const role = msg.role.charAt(0).toUpperCase() + msg.role.slice(1);
            prompt += `${role}: ${msg.content}\n`;
        }
        prompt += "Assistant: ";
        const completion = await this.generateTextCompletion(prompt, options);
        const generatedText = completion.choices[0].text;
        return {
            id: "chatcmpl-" + Math.random().toString(36).substring(2),
            object: "chat.completion",
            created: Math.floor(Date.now() / 1000),
            model: this.config.model_name || "LlamaModel",
            choices: [
                {
                    index: 0,
                    message: { role: "assistant", content: generatedText },
                    finish_reason: "stop"
                }
            ]
        };
    }
}

// module.exports = LlamaModel;

// Define the generateReply function
async function generateReply(messages) {
    const model = new LlamaModel({ model_name: 'LlamaModel' }, window.device); // Use the globally accessible device
    const response = await model.generateChatCompletion(messages);
    return response.choices[0].message.content;
}

// Export both LlamaModel and generateReply
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { 
        LlamaModel, 
        generateReply 
    };
}