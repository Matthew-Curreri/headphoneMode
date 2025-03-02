// Transformer model implementation with vanilla JavaScript and WebGPU API
// This code mirrors a PyTorch implementation, replacing tensor ops with WebGPU compute shaders.

async function TransformerCreate(config) {
    // Default configuration values
    const defaultConfig = {
        d_model: 512,
        n_heads: 8,
        d_ff: 2048,
        n_layers: 4,
        vocab_size: 32000,
        max_seq_len: 1024,
        epsilon: 1e-5  // RMSNorm epsilon
    };
    // Merge provided config with defaults
    const cfg = Object.assign({}, defaultConfig, config);

    // Derived parameters
    cfg.head_dim = cfg.d_model / cfg.n_heads;
    if (!Number.isInteger(cfg.head_dim)) {
        throw new Error("d_model must be divisible by n_heads");
    }

    // Initialize WebGPU device
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("WebGPU adapter not found");
    const device = await adapter.requestDevice();

    // Precompute rotary frequency embedding tables&#8203;:contentReference[oaicite:0]{index=0}
    const halfDim = cfg.head_dim / 2;
    const invFreq = new Float32Array(halfDim);
    for (let i = 0; i < halfDim; i++) {
        invFreq[i] = 1.0 / Math.pow(10000.0, (2 * i) / cfg.head_dim);
    }
    const maxSeq = cfg.max_seq_len;
    const cosData = new Float32Array(maxSeq * halfDim);
    const sinData = new Float32Array(maxSeq * halfDim);
    for (let pos = 0; pos < maxSeq; pos++) {
        for (let i = 0; i < halfDim; i++) {
            const angle = pos * invFreq[i];
            cosData[pos * halfDim + i] = Math.cos(angle);
            sinData[pos * halfDim + i] = Math.sin(angle);
        }
    }
    // Create GPU buffers for precomputed cos and sin tables (size: max_seq_len x (head_dim/2))
    const cosBuffer = device.createBuffer({
        mappedAtCreation: true,
        size: (2 + cosData.length) * 4,  // 2 floats for shape + data
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    {
        const map = new Float32Array(cosBuffer.getMappedRange());
        // Store dimensions (max_seq_len, halfDim) followed by data
        map[0] = maxSeq;
        map[1] = halfDim;
        map.set(cosData, 2);
        cosBuffer.unmap();
    }
    const sinBuffer = device.createBuffer({
        mappedAtCreation: true,
        size: (2 + sinData.length) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    {
        const map = new Float32Array(sinBuffer.getMappedRange());
        map[0] = maxSeq;
        map[1] = halfDim;
        map.set(sinData, 2);
        sinBuffer.unmap();
    }

    // Allocate and initialize model weights
    // Note: We use small random values for weights. In practice, load pretrained weights or initialize properly.
    function randn(scale = 0.02) {
        return (Math.random() * 2 - 1) * scale;
    }
    // Embedding matrix: vocab_size x d_model
    const embedSize = cfg.vocab_size * cfg.d_model;
    const embedWeights = new Float32Array(embedSize);
    for (let i = 0; i < embedSize; i++) {
        embedWeights[i] = randn();
    }
    const embedBuffer = device.createBuffer({
        mappedAtCreation: true,
        size: (2 + embedWeights.length) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    {
        const map = new Float32Array(embedBuffer.getMappedRange());
        map[0] = cfg.vocab_size;
        map[1] = cfg.d_model;
        map.set(embedWeights, 2);
        embedBuffer.unmap();
    }

    // Weight arrays for each transformer layer
    const norm1Buffers = [];
    const norm2Buffers = [];
    const WqBuffers = [];  // WqBuffers[layer][head]
    const WkBuffers = [];
    const WvBuffers = [];
    const WoBuffers = [];  // Wo segmented by heads per layer
    const W1Buffers = [];
    const W2Buffers = [];
    for (let l = 0; l < cfg.n_layers; l++) {
        // RMSNorm weight vectors (initialized to 1.0)
        const norm1 = new Float32Array(cfg.d_model);
        const norm2 = new Float32Array(cfg.d_model);
        norm1.fill(1.0);
        norm2.fill(1.0);
        const norm1Buf = device.createBuffer({
            mappedAtCreation: true,
            size: (2 + norm1.length) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        const norm2Buf = device.createBuffer({
            mappedAtCreation: true,
            size: (2 + norm2.length) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        {
            const m1 = new Float32Array(norm1Buf.getMappedRange());
            m1[0] = 1; m1[1] = cfg.d_model;
            m1.set(norm1, 2);
            norm1Buf.unmap();
        }
        {
            const m2 = new Float32Array(norm2Buf.getMappedRange());
            m2[0] = 1; m2[1] = cfg.d_model;
            m2.set(norm2, 2);
            norm2Buf.unmap();
        }
        norm1Buffers.push(norm1Buf);
        norm2Buffers.push(norm2Buf);

        // Attention weights: Wq, Wk, Wv for each head, and output Wo segments
        WqBuffers[l] = [];
        WkBuffers[l] = [];
        WvBuffers[l] = [];
        WoBuffers[l] = [];
        for (let h = 0; h < cfg.n_heads; h++) {
            const wq = new Float32Array(cfg.d_model * cfg.head_dim);
            const wk = new Float32Array(cfg.d_model * cfg.head_dim);
            const wv = new Float32Array(cfg.d_model * cfg.head_dim);
            for (let i = 0; i < wq.length; i++) {
                wq[i] = randn();
                wk[i] = randn();
                wv[i] = randn();
            }
            const wqBuf = device.createBuffer({
                mappedAtCreation: true,
                size: (2 + wq.length) * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            });
            const wkBuf = device.createBuffer({
                mappedAtCreation: true,
                size: (2 + wk.length) * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            });
            const wvBuf = device.createBuffer({
                mappedAtCreation: true,
                size: (2 + wv.length) * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            });
            {
                const mq = new Float32Array(wqBuf.getMappedRange());
                mq[0] = cfg.d_model;
                mq[1] = cfg.head_dim;
                mq.set(wq, 2);
                wqBuf.unmap();
            }
            {
                const mk = new Float32Array(wkBuf.getMappedRange());
                mk[0] = cfg.d_model;
                mk[1] = cfg.head_dim;
                mk.set(wk, 2);
                wkBuf.unmap();
            }
            {
                const mv = new Float32Array(wvBuf.getMappedRange());
                mv[0] = cfg.d_model;
                mv[1] = cfg.head_dim;
                mv.set(wv, 2);
                wvBuf.unmap();
            }
            WqBuffers[l].push(wqBuf);
            WkBuffers[l].push(wkBuf);
            WvBuffers[l].push(wvBuf);

            // Wo segment for this head (shape: head_dim x d_model)
            const wo = new Float32Array(cfg.head_dim * cfg.d_model);
            for (let i = 0; i < wo.length; i++) wo[i] = randn();
            const woBuf = device.createBuffer({
                mappedAtCreation: true,
                size: (2 + wo.length) * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            });
            {
                const mo = new Float32Array(woBuf.getMappedRange());
                mo[0] = cfg.head_dim;
                mo[1] = cfg.d_model;
                mo.set(wo, 2);
                woBuf.unmap();
            }
            WoBuffers[l].push(woBuf);
        }

        // Feed-forward weights: W1 (d_model x d_ff) and W2 (d_ff x d_model)
        const w1 = new Float32Array(cfg.d_model * cfg.d_ff);
        const w2 = new Float32Array(cfg.d_ff * cfg.d_model);
        for (let i = 0; i < w1.length; i++) {
            w1[i] = randn();
        }
        for (let j = 0; j < w2.length; j++) {
            w2[j] = randn();
        }
        const w1Buf = device.createBuffer({
            mappedAtCreation: true,
            size: (2 + w1.length) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        const w2Buf = device.createBuffer({
            mappedAtCreation: true,
            size: (2 + w2.length) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        {
            const m1 = new Float32Array(w1Buf.getMappedRange());
            m1[0] = cfg.d_model;
            m1[1] = cfg.d_ff;
            m1.set(w1, 2);
            w1Buf.unmap();
        }
        {
            const m2 = new Float32Array(w2Buf.getMappedRange());
            m2[0] = cfg.d_ff;
            m2[1] = cfg.d_model;
            m2.set(w2, 2);
            w2Buf.unmap();
        }
        W1Buffers.push(w1Buf);
        W2Buffers.push(w2Buf);
    }

    // Forward pass: run the model on a given sequence of input token IDs
    async function forward(inputIds) {
        const seqLen = inputIds.length;
        if (seqLen > cfg.max_seq_len) throw new Error("Input sequence too long");
        // Prepare input indices buffer
        const indicesBuffer = device.createBuffer({
            mappedAtCreation: true,
            size: (2 + seqLen) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        {
            const map = new Float32Array(indicesBuffer.getMappedRange());
            map[0] = 1;  // treat indices as 1 x seqLen matrix
            map[1] = seqLen;
            for (let i = 0; i < seqLen; i++) {
                map[2 + i] = inputIds[i];
            }
            indicesBuffer.unmap();
        }
        // Buffer for current hidden state (seqLen x d_model)
        const xBuffer = device.createBuffer({
            mappedAtCreation: true,
            size: (2 + seqLen * cfg.d_model) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        {
            // Set initial hidden state size metadata
            const map = new Float32Array(xBuffer.getMappedRange());
            map[0] = seqLen;
            map[1] = cfg.d_model;
            xBuffer.unmap();
        }
        // Compute embedding lookup
        const bindGroup = device.createBindGroup({
            layout: embedPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: embedBuffer } },
                { binding: 1, resource: { buffer: indicesBuffer } },
                { binding: 2, resource: { buffer: xBuffer } }
            ]
        });
        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(embedPipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(Math.ceil(seqLen / 16), Math.ceil(cfg.d_model / 16));

        // For each transformer layer
        for (let l = 0; l < cfg.n_layers; l++) {
            // RMSNorm (attention input)
            const norm1OutBuf = device.createBuffer({
                mappedAtCreation: true,
                size: (2 + seqLen * cfg.d_model) * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            });
            {
                const mn = new Float32Array(norm1OutBuf.getMappedRange());
                mn[0] = seqLen;
                mn[1] = cfg.d_model;
                norm1OutBuf.unmap();
            }
            {
                const bindGroupNorm1 = device.createBindGroup({
                    layout: rmsNormPipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: xBuffer } },
                        { binding: 1, resource: { buffer: norm1OutBuf } },
                        { binding: 2, resource: { buffer: norm1Buffers[l] } }
                    ]
                });
                passEncoder.setPipeline(rmsNormPipeline);
                passEncoder.setBindGroup(0, bindGroupNorm1);
                passEncoder.dispatchWorkgroups(seqLen);
            }

            // Prepare buffers for attention outputs
            const attnOutBuf = device.createBuffer({
                mappedAtCreation: true,
                size: (2 + seqLen * cfg.d_model) * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            });
            {
                const ma = new Float32Array(attnOutBuf.getMappedRange());
                ma[0] = seqLen;
                ma[1] = cfg.d_model;
                attnOutBuf.unmap();
            }
            const partBuf = device.createBuffer({
                mappedAtCreation: true,
                size: (2 + seqLen * cfg.d_model) * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            });
            {
                const mp = new Float32Array(partBuf.getMappedRange());
                mp[0] = seqLen;
                mp[1] = cfg.d_model;
                partBuf.unmap();
            }

            // Multi-head self-attention
            for (let h = 0; h < cfg.n_heads; h++) {
                // Compute Q, K, V for head h
                const QBuffer = device.createBuffer({
                    mappedAtCreation: true,
                    size: (2 + seqLen * cfg.head_dim) * 4,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
                });
                {
                    const mq = new Float32Array(QBuffer.getMappedRange());
                    mq[0] = seqLen;
                    mq[1] = cfg.head_dim;
                    QBuffer.unmap();
                }
                const KBuffer = device.createBuffer({
                    mappedAtCreation: true,
                    size: (2 + seqLen * cfg.head_dim) * 4,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
                });
                {
                    const mk = new Float32Array(KBuffer.getMappedRange());
                    mk[0] = seqLen;
                    mk[1] = cfg.head_dim;
                    KBuffer.unmap();
                }
                const VBuffer = device.createBuffer({
                    mappedAtCreation: true,
                    size: (2 + seqLen * cfg.head_dim) * 4,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
                });
                {
                    const mv = new Float32Array(VBuffer.getMappedRange());
                    mv[0] = seqLen;
                    mv[1] = cfg.head_dim;
                    VBuffer.unmap();
                }
                // Q_h = norm1OutBuf (seqLen x d_model) * Wq[l][h] (d_model x head_dim)
                {
                    const bindGroupQ = device.createBindGroup({
                        layout: matMulPipeline.getBindGroupLayout(0),
                        entries: [
                            { binding: 0, resource: { buffer: norm1OutBuf } },
                            { binding: 1, resource: { buffer: WqBuffers[l][h] } },
                            { binding: 2, resource: { buffer: QBuffer } }
                        ]
                    });
                    passEncoder.setPipeline(matMulPipeline);
                    passEncoder.setBindGroup(0, bindGroupQ);
                    passEncoder.dispatchWorkgroups(Math.ceil(seqLen / 8), Math.ceil(cfg.head_dim / 8));
                }
                // K_h = norm1OutBuf * Wk[l][h]
                {
                    const bindGroupK = device.createBindGroup({
                        layout: matMulPipeline.getBindGroupLayout(0),
                        entries: [
                            { binding: 0, resource: { buffer: norm1OutBuf } },
                            { binding: 1, resource: { buffer: WkBuffers[l][h] } },
                            { binding: 2, resource: { buffer: KBuffer } }
                        ]
                    });
                    passEncoder.setPipeline(matMulPipeline);
                    passEncoder.setBindGroup(0, bindGroupK);
                    passEncoder.dispatchWorkgroups(Math.ceil(seqLen / 8), Math.ceil(cfg.head_dim / 8));
                }
                // V_h = norm1OutBuf * Wv[l][h]
                {
                    const bindGroupV = device.createBindGroup({
                        layout: matMulPipeline.getBindGroupLayout(0),
                        entries: [
                            { binding: 0, resource: { buffer: norm1OutBuf } },
                            { binding: 1, resource: { buffer: WvBuffers[l][h] } },
                            { binding: 2, resource: { buffer: VBuffer } }
                        ]
                    });
                    passEncoder.setPipeline(matMulPipeline);
                    passEncoder.setBindGroup(0, bindGroupV);
                    passEncoder.dispatchWorkgroups(Math.ceil(seqLen / 8), Math.ceil(cfg.head_dim / 8));
                }
                // Apply rotary positional embeddings to Q_h and K_h
                {
                    const bindGroupRot = device.createBindGroup({
                        layout: rotaryPipeline.getBindGroupLayout(0),
                        entries: [
                            { binding: 0, resource: { buffer: QBuffer } },
                            { binding: 1, resource: { buffer: KBuffer } },
                            { binding: 2, resource: { buffer: cosBuffer } },
                            { binding: 3, resource: { buffer: sinBuffer } }
                        ]
                    });
                    passEncoder.setPipeline(rotaryPipeline);
                    passEncoder.setBindGroup(0, bindGroupRot);
                    passEncoder.dispatchWorkgroups(Math.ceil(seqLen / 8), Math.ceil((cfg.head_dim / 2) / 8));
                }
                // Compute attention scores = Q_h * K_h^T (with causal mask)
                const scoresBuffer = device.createBuffer({
                    mappedAtCreation: true,
                    size: (2 + seqLen * seqLen) * 4,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
                });
                {
                    const ms = new Float32Array(scoresBuffer.getMappedRange());
                    ms[0] = seqLen;
                    ms[1] = seqLen;
                    scoresBuffer.unmap();
                }
                {
                    const bindGroupScore = device.createBindGroup({
                        layout: attnScorePipeline.getBindGroupLayout(0),
                        entries: [
                            { binding: 0, resource: { buffer: QBuffer } },
                            { binding: 1, resource: { buffer: KBuffer } },
                            { binding: 2, resource: { buffer: scoresBuffer } }
                        ]
                    });
                    passEncoder.setPipeline(attnScorePipeline);
                    passEncoder.setBindGroup(0, bindGroupScore);
                    passEncoder.dispatchWorkgroups(Math.ceil(seqLen / 8), Math.ceil(seqLen / 8));
                }
                // Softmax over attention scores (row-wise)
                {
                    const bindGroupSoftmax = device.createBindGroup({
                        layout: softmaxPipeline.getBindGroupLayout(0),
                        entries: [
                            { binding: 0, resource: { buffer: scoresBuffer } }
                        ]
                    });
                    passEncoder.setPipeline(softmaxPipeline);
                    passEncoder.setBindGroup(0, bindGroupSoftmax);
                    passEncoder.dispatchWorkgroups(seqLen);
                }
                // Compute context = softmax(scores) * V_h  -> (seqLen x head_dim)
                const contextBuffer = device.createBuffer({
                    mappedAtCreation: true,
                    size: (2 + seqLen * cfg.head_dim) * 4,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
                });
                {
                    const mc = new Float32Array(contextBuffer.getMappedRange());
                    mc[0] = seqLen;
                    mc[1] = cfg.head_dim;
                    contextBuffer.unmap();
                }
                {
                    const bindGroupContext = device.createBindGroup({
                        layout: matMulPipeline.getBindGroupLayout(0),
                        entries: [
                            { binding: 0, resource: { buffer: scoresBuffer } },
                            { binding: 1, resource: { buffer: VBuffer } },
                            { binding: 2, resource: { buffer: contextBuffer } }
                        ]
                    });
                    passEncoder.setPipeline(matMulPipeline);
                    passEncoder.setBindGroup(0, bindGroupContext);
                    passEncoder.dispatchWorkgroups(Math.ceil(seqLen / 8), Math.ceil(cfg.head_dim / 8));
                }
                // Project context through output matrix W_o segment
                if (h === 0) {
                    // First head: write directly to attnOutBuf
                    const bindGroupOut0 = device.createBindGroup({
                        layout: matMulPipeline.getBindGroupLayout(0),
                        entries: [
                            { binding: 0, resource: { buffer: contextBuffer } },
                            { binding: 1, resource: { buffer: WoBuffers[l][h] } },
                            { binding: 2, resource: { buffer: attnOutBuf } }
                        ]
                    });
                    passEncoder.setPipeline(matMulPipeline);
                    passEncoder.setBindGroup(0, bindGroupOut0);
                    passEncoder.dispatchWorkgroups(Math.ceil(seqLen / 8), Math.ceil(cfg.d_model / 8));
                } else {
                    const bindGroupOut = device.createBindGroup({
                        layout: matMulPipeline.getBindGroupLayout(0),
                        entries: [
                            { binding: 0, resource: { buffer: contextBuffer } },
                            { binding: 1, resource: { buffer: WoBuffers[l][h] } },
                            { binding: 2, resource: { buffer: partBuf } }
                        ]
                    });
                    passEncoder.setPipeline(matMulPipeline);
                    passEncoder.setBindGroup(0, bindGroupOut);
                    passEncoder.dispatchWorkgroups(Math.ceil(seqLen / 8), Math.ceil(cfg.d_model / 8));
                    // Add partial output into attnOutBuf
                    const bindGroupAdd = device.createBindGroup({
                        layout: residualAddPipeline.getBindGroupLayout(0),
                        entries: [
                            { binding: 0, resource: { buffer: attnOutBuf } },
                            { binding: 1, resource: { buffer: partBuf } }
                        ]
                    });
                    passEncoder.setPipeline(residualAddPipeline);
                    passEncoder.setBindGroup(0, bindGroupAdd);
                    passEncoder.dispatchWorkgroups(Math.ceil(seqLen / 16), Math.ceil(cfg.d_model / 16));
                }
            }
            // Add attention output to residual (update xBuffer)
            {
                const bindGroupResAttn = device.createBindGroup({
                    layout: residualAddPipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: xBuffer } },
                        { binding: 1, resource: { buffer: attnOutBuf } }
                    ]
                });
                passEncoder.setPipeline(residualAddPipeline);
                passEncoder.setBindGroup(0, bindGroupResAttn);
                passEncoder.dispatchWorkgroups(Math.ceil(seqLen / 16), Math.ceil(cfg.d_model / 16));
            }

            // RMSNorm (feed-forward input)
            const norm2OutBuf = device.createBuffer({
                mappedAtCreation: true,
                size: (2 + seqLen * cfg.d_model) * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            });
            {
                const mn2 = new Float32Array(norm2OutBuf.getMappedRange());
                mn2[0] = seqLen;
                mn2[1] = cfg.d_model;
                norm2OutBuf.unmap();
            }
            {
                const bindGroupNorm2 = device.createBindGroup({
                    layout: rmsNormPipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: xBuffer } },
                        { binding: 1, resource: { buffer: norm2OutBuf } },
                        { binding: 2, resource: { buffer: norm2Buffers[l] } }
                    ]
                });
                passEncoder.setPipeline(rmsNormPipeline);
                passEncoder.setBindGroup(0, bindGroupNorm2);
                passEncoder.dispatchWorkgroups(seqLen);
            }

            // Feed-forward network
            const ff1Buf = device.createBuffer({
                mappedAtCreation: true,
                size: (2 + seqLen * cfg.d_ff) * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            });
            {
                const mf1 = new Float32Array(ff1Buf.getMappedRange());
                mf1[0] = seqLen;
                mf1[1] = cfg.d_ff;
                ff1Buf.unmap();
            }
            {
                const bindGroupFF1 = device.createBindGroup({
                    layout: matMulPipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: norm2OutBuf } },
                        { binding: 1, resource: { buffer: W1Buffers[l] } },
                        { binding: 2, resource: { buffer: ff1Buf } }
                    ]
                });
                passEncoder.setPipeline(matMulPipeline);
                passEncoder.setBindGroup(0, bindGroupFF1);
                passEncoder.dispatchWorkgroups(Math.ceil(seqLen / 8), Math.ceil(cfg.d_ff / 8));
            }
            // Activation (GeLU) on ff1Buf
            {
                const bindGroupGelu = device.createBindGroup({
                    layout: geluPipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: ff1Buf } }
                    ]
                });
                passEncoder.setPipeline(geluPipeline);
                passEncoder.setBindGroup(0, bindGroupGelu);
                passEncoder.dispatchWorkgroups(Math.ceil(seqLen / 8), Math.ceil(cfg.d_ff / 8));
            }
            const ff2Buf = device.createBuffer({
                mappedAtCreation: true,
                size: (2 + seqLen * cfg.d_model) * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            });
            {
                const mf2 = new Float32Array(ff2Buf.getMappedRange());
                mf2[0] = seqLen;
                mf2[1] = cfg.d_model;
                ff2Buf.unmap();
            }
            {
                const bindGroupFF2 = device.createBindGroup({
                    layout: matMulPipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: ff1Buf } },
                        { binding: 1, resource: { buffer: W2Buffers[l] } },
                        { binding: 2, resource: { buffer: ff2Buf } }
                    ]
                });
                passEncoder.setPipeline(matMulPipeline);
                passEncoder.setBindGroup(0, bindGroupFF2);
                passEncoder.dispatchWorkgroups(Math.ceil(seqLen / 8), Math.ceil(cfg.d_model / 8));
            }
            // Add feed-forward output to residual (update xBuffer)
            {
                const bindGroupResFF = device.createBindGroup({
                    layout: residualAddPipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: xBuffer } },
                        { binding: 1, resource: { buffer: ff2Buf } }
                    ]
                });
                passEncoder.setPipeline(residualAddPipeline);
                passEncoder.setBindGroup(0, bindGroupResFF);
                passEncoder.dispatchWorkgroups(Math.ceil(seqLen / 16), Math.ceil(cfg.d_model / 16));
            }
        }

        // Final RMSNorm on output
        const outNormBuf = device.createBuffer({
            mappedAtCreation: true,
            size: (2 + seqLen * cfg.d_model) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        {
            const mo = new Float32Array(outNormBuf.getMappedRange());
            mo[0] = seqLen;
            mo[1] = cfg.d_model;
            outNormBuf.unmap();
        }
        {
            const bindGroupOutNorm = device.createBindGroup({
                layout: rmsNormPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: xBuffer } },
                    { binding: 1, resource: { buffer: outNormBuf } },
                    { binding: 2, resource: { buffer: norm1Buffers[cfg.n_layers - 1] } }
                ]
            });
            passEncoder.setPipeline(rmsNormPipeline);
            passEncoder.setBindGroup(0, bindGroupOutNorm);
            passEncoder.dispatchWorkgroups(seqLen);
        }

        // End compute pass and submit commands
        passEncoder.end();
        const commands = commandEncoder.finish();
        device.queue.submit([commands]);

        // Copy output data from GPU and return as Float32Array
        const outReadBuf = device.createBuffer({
            size: seqLen * cfg.d_model * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        {
            const copyEncoder = device.createCommandEncoder();
            copyEncoder.copyBufferToBuffer(outNormBuf, 8 /* skip size header */, outReadBuf, 0, seqLen * cfg.d_model * 4);
            device.queue.submit([copyEncoder.finish()]);
        }
        await outReadBuf.mapAsync(GPUMapMode.READ);
        const arrayBuffer = outReadBuf.getMappedRange();
        const outputArray = new Float32Array(arrayBuffer.slice(0));
        outReadBuf.unmap();
        return outputArray;
    } // end of forward

    // Return an object with the forward function (and possibly device for further use)
    return {
        config: cfg,
        device: device,
        forward: forward
    };
} // end of TransformerCreate

// Set up WebGPU compute pipelines (WGSL shaders)
const shaderModuleMatMul = device.createShaderModule({ code: `
    struct Matrix { size: vec2<f32>; numbers: array<f32>; };
    @group(0) @binding(0) var<storage, read> A: Matrix;
    @group(0) @binding(1) var<storage, read> B: Matrix;
    @group(0) @binding(2) var<storage, read_write> C: Matrix;
    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) global_id: vec3u) {
        let rows = u32(A.size.x);
        let inner = u32(A.size.y);
        let cols = u32(B.size.y);
        let i = global_id.x;
        let j = global_id.y;
        if (i >= rows || j >= cols) {
            return;
        }
        var sum = 0.0;
        for (var k: u32 = 0u; k < inner; k = k + 1u) {
            let aIndex = k + i * inner;
            let bIndex = j + k * cols;
            sum = sum + A.numbers[aIndex] * B.numbers[bIndex];
        }
        let idx = j + i * cols;
        C.numbers[idx] = sum;
    }
` });
const matMulPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: shaderModuleMatMul, entryPoint: "main" }
});

const shaderModuleAttnScore = device.createShaderModule({ code: `
    struct Matrix { size: vec2<f32>; numbers: array<f32>; };
    @group(0) @binding(0) var<storage, read> Q : Matrix;
    @group(0) @binding(1) var<storage, read> K : Matrix;
    @group(0) @binding(2) var<storage, read_write> Scores : Matrix;
    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) global_id: vec3u) {
        let seqLen = u32(Q.size.x);
        let i = global_id.x;
        let j = global_id.y;
        if (i >= seqLen || j >= seqLen) {
            return;
        }
        let headDim = u32(Q.size.y);
        let idx = j + i * seqLen;
        if (j > i) {
            Scores.numbers[idx] = -1e9;
            return;
        }
        var sum = 0.0;
        for (var k: u32 = 0u; k < headDim; k = k + 1u) {
            let qIndex = k + i * headDim;
            let kIndex = k + j * headDim;
            sum = sum + Q.numbers[qIndex] * K.numbers[kIndex];
        }
        sum = sum / sqrt(f32(headDim));
        Scores.numbers[idx] = sum;
    }
` });
const attnScorePipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: shaderModuleAttnScore, entryPoint: "main" }
});

const shaderModuleSoftmax = device.createShaderModule({ code: `
    struct Matrix { size: vec2<f32>; numbers: array<f32>; };
    @group(0) @binding(0) var<storage, read_write> Scores : Matrix;
    @compute @workgroup_size(1)
    fn main(@builtin(global_invocation_id) global_id: vec3u) {
        let seqLen = u32(Scores.size.x);
        let i = global_id.x;
        if (i >= seqLen) { return; }
        let cols = u32(Scores.size.y);
        var maxVal = -1e30;
        // find max for numerical stability
        for (var j: u32 = 0u; j < cols; j = j + 1u) {
            let idx = j + i * cols;
            let v = Scores.numbers[idx];
            if (v > maxVal) { maxVal = v; }
        }
        var sum = 0.0;
        // compute exp and sum
        for (var j: u32 = 0u; j < cols; j = j + 1u) {
            let idx = j + i * cols;
            let expVal = exp(Scores.numbers[idx] - maxVal);
            Scores.numbers[idx] = expVal;
            sum = sum + expVal;
        }
        // normalize
        for (var j: u32 = 0u; j < cols; j = j + 1u) {
            let idx = j + i * cols;
            Scores.numbers[idx] = Scores.numbers[idx] / sum;
        }
    }
` });
const softmaxPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: shaderModuleSoftmax, entryPoint: "main" }
});

const shaderModuleRMSNorm = device.createShaderModule({ code: `
    struct Matrix { size: vec2<f32>; numbers: array<f32>; };
    @group(0) @binding(0) var<storage, read> X : Matrix;
    @group(0) @binding(1) var<storage, read_write> Y : Matrix;
    @group(0) @binding(2) var<storage, read> Weight : Matrix;
    @compute @workgroup_size(1)
    fn main(@builtin(global_invocation_id) global_id: vec3u) {
        let n = u32(X.size.x);
        let d = u32(X.size.y);
        let i = global_id.x;
        if (i >= n) { return; }
        var sumSquares = 0.0;
        for (var j: u32 = 0u; j < d; j = j + 1u) {
            let idx = j + i * d;
            let val = X.numbers[idx];
            sumSquares = sumSquares + val * val;
        }
        let meanSq = sumSquares / f32(d);
        let invRms = 1.0 / sqrt(meanSq + ${cfg.epsilon});
        for (var j: u32 = 0u; j < d; j = j + 1u) {
            let idx = j + i * d;
            Y.numbers[idx] = X.numbers[idx] * invRms * Weight.numbers[j];
        }
    }
` });
const rmsNormPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: shaderModuleRMSNorm, entryPoint: "main" }
});

const shaderModuleResidualAdd = device.createShaderModule({ code: `
    struct Matrix { size: vec2<f32>; numbers: array<f32>; };
    @group(0) @binding(0) var<storage, read_write> Y : Matrix;
    @group(0) @binding(1) var<storage, read> X : Matrix;
    @compute @workgroup_size(16, 16)
    fn main(@builtin(global_invocation_id) global_id: vec3u) {
        let rows = u32(Y.size.x);
        let cols = u32(Y.size.y);
        let i = global_id.x;
        let j = global_id.y;
        if (i < rows && j < cols) {
            let idx = j + i * cols;
            Y.numbers[idx] = Y.numbers[idx] + X.numbers[idx];
        }
    }
` });
const residualAddPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: shaderModuleResidualAdd, entryPoint: "main" }
});

const shaderModuleGelu = device.createShaderModule({ code: `
    struct Matrix { size: vec2<f32>; numbers: array<f32>; };
    fn tanh_custom(x: f32) -> f32 {
        let e = exp(2.0 * x);
        return (e - 1.0) / (e + 1.0);
    }
    @group(0) @binding(0) var<storage, read_write> X : Matrix;
    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) global_id: vec3u) {
        let rows = u32(X.size.x);
        let cols = u32(X.size.y);
        let i = global_id.x;
        let j = global_id.y;
        if (i >= rows || j >= cols) { return; }
        let idx = j + i * cols;
        let val = X.numbers[idx];
        // GELU approximation
        let u = 0.797884 * (val + 0.044715 * val * val * val);
        let t = tanh_custom(u);
        X.numbers[idx] = 0.5 * val * (1.0 + t);
    }
` });
const geluPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: shaderModuleGelu, entryPoint: "main" }
});

const shaderModuleRotary = device.createShaderModule({ code: `
    struct Matrix { size: vec2<f32>; numbers: array<f32>; };
    @group(0) @binding(0) var<storage, read_write> Q : Matrix;
    @group(0) @binding(1) var<storage, read_write> K : Matrix;
    @group(0) @binding(2) var<storage, read> Cos : Matrix;
    @group(0) @binding(3) var<storage, read> Sin : Matrix;
    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) global_id: vec3u) {
        let seqLen = u32(Q.size.x);
        let halfDim = u32(Cos.size.y);
        let i = global_id.x;
        let j = global_id.y;
        if (i >= seqLen || j >= halfDim) { return; }
        let idx1 = 2u * j + i * u32(Q.size.y);
        let idx2 = (2u * j + 1u) + i * u32(Q.size.y);
        let cosVal = Cos.numbers[j + i * halfDim];
        let sinVal = Sin.numbers[j + i * halfDim];
        let q1 = Q.numbers[idx1];
        let q2 = Q.numbers[idx2];
        let k1 = K.numbers[idx1];
        let k2 = K.numbers[idx2];
        Q.numbers[idx1] = q1 * cosVal - q2 * sinVal;
        Q.numbers[idx2] = q1 * sinVal + q2 * cosVal;
        K.numbers[idx1] = k1 * cosVal - k2 * sinVal;
        K.numbers[idx2] = k1 * sinVal + k2 * cosVal;
    }
` });
const rotaryPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: shaderModuleRotary, entryPoint: "main" }
});

const shaderModuleEmbed = device.createShaderModule({ code: `
    struct Matrix { size: vec2<f32>; numbers: array<f32>; };
    @group(0) @binding(0) var<storage, read> Embed: Matrix;
    @group(0) @binding(1) var<storage, read> Indices: Matrix;
    @group(0) @binding(2) var<storage, read_write> Output: Matrix;
    @compute @workgroup_size(16, 16)
    fn main(@builtin(global_invocation_id) global_id: vec3u) {
        let seqLen = u32(Output.size.x);
        let dmodel = u32(Output.size.y);
        let i = global_id.x;
        let j = global_id.y;
        if (i < seqLen && j < dmodel) {
            let tokenId = u32(Indices.numbers[i]);
            let embedIdx = j + tokenId * dmodel;
            let outIdx = j + i * dmodel;
            Output.numbers[outIdx] = Embed.numbers[embedIdx];
        }
    }
` });
const embedPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: shaderModuleEmbed, entryPoint: "main" }
});
