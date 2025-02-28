//Use gpu ui to off load both llm and audio processing onto the client.
// Assume gamma is a learned weight vector of shape [d_model]
function rmsNorm(x, gamma, epsilon = 1e-6) {
    // x shape: [batch, seq_len, d_model] (or [seq_len, d_model] if batch size 1 for simplicity)
    // Compute mean of squares for each element vector along last dimension:
    const meanSquares = tf.mean(tf.square(x), /*axis=*/-1, /*keepDims=*/true);
    const invRms = tf.rsqrt(meanSquares.add(epsilon));  // 1/sqrt(meanSquares + eps)
    const normalized = x.mul(invRms);                   // scale input by invRMS
    return normalized.mul(gamma);  // apply gain (gamma vector broadcasted along seq_len)
  }
  // Weight matrices: Wq, Wk, Wv of shape [d_model, d_model] and Wo of shape [d_model, d_model]
// Assuming `nHeads` heads and each head has dimension d_head = d_model / nHeads
function multiHeadAttention(x, Wq, Wk, Wv, Wo, nHeads, rotarySin, rotaryCos) {
    // 1. Linear projections for Q, K, V
    const q = tf.matMul(x, Wq);  // shape: [batch, seq_len, d_model]
    const k = tf.matMul(x, Wk);
    const v = tf.matMul(x, Wv);
  
    // 2. Reshape Q, K, V to [batch, seq_len, nHeads, d_head] for multi-head
    const dModel = Wq.shape[0], dHead = dModel / nHeads;
    const newShape = [-1, tf.shape(x).arraySync()[1], nHeads, dHead];  // [batch, seq_len, nHeads, d_head]
    // (Note: tf.shape(x).arraySync() is not ideal inside a graph; in real code we'd get seq_len differently.
    // For simplicity, assume known seq_len or use tf.reshape with explicit constants.)
    const Q = tf.reshape(q, newShape);
    const K = tf.reshape(k, newShape);
    const V = tf.reshape(v, newShape);
  
    // 3. Apply rotary embeddings to Q and K.
    // rotarySin and rotaryCos are precomputed tensors of shape [seq_len, d_head] (or [1, seq_len, d_head] to broadcast).
    // They contain sine and cosine values for each position and each pair of dimensions.
    const [Q_rot, K_rot] = applyRotaryEmbedding(Q, K, rotarySin, rotaryCos);
  
    // 4. Scaled dot-product attention for each head.
    // We need Q*K^T for each head. We can use batch matmul by merging batch and head dims.
    const Q2 = tf.transpose(Q_rot, [0, 2, 1, 3]);  // shape [batch, nHeads, seq_len, d_head]
    const K2 = tf.transpose(K_rot, [0, 2, 1, 3]);  // shape [batch, nHeads, seq_len, d_head]
    // Now treat batch*nHeads as one combined batch for matmul:
    const batchHeads = -1;  // effectively batch * nHeads
    const Q_flat = tf.reshape(Q2, [batchHeads, tf.shape(Q2).arraySync()[2], dHead]);  // [batch*nHeads, seq_len, d_head]
    const K_flat = tf.reshape(K2, [batchHeads, tf.shape(K2).arraySync()[2], dHead]);  // [batch*nHeads, seq_len, d_head]
    // Compute attention scores: [batch*nHeads, seq_len, seq_len]
    let attnScores = tf.matMul(Q_flat, K_flat, false, true);  // K^T via transpose_b=true
    // Scale scores by sqrt(d_head):
    attnScores = attnScores.div(Math.sqrt(dHead));
    // Softmax along the last axis (keys length):
    attnScores = tf.softmax(attnScores, -1);
  
    // 5. Use scores to weighted sum the values.
    const V_flat = tf.reshape(tf.transpose(V, [0, 2, 1, 3]), [batchHeads, tf.shape(V).arraySync()[1], dHead]);
    // [batch*nHeads, seq_len, d_head]
    let attnOutput = tf.matMul(attnScores, V_flat);  // [batch*nHeads, seq_len, d_head]
    // Reshape back to [batch, nHeads, seq_len, d_head] then transpose to [batch, seq_len, nHeads, d_head]
    attnOutput = tf.reshape(attnOutput, [ -1, nHeads, tf.shape(x).arraySync()[1], dHead ]);
    attnOutput = tf.transpose(attnOutput, [0, 2, 1, 3]);  // [batch, seq_len, nHeads, d_head]
    // 6. Concatenate heads (merge nHeads and d_head back into d_model)
    const concatOutput = tf.reshape(attnOutput, [-1, tf.shape(x).arraySync()[1], dModel]);
  
    // 7. Final linear projection
    const result = tf.matMul(concatOutput, Wo);
    return result;  // [batch, seq_len, d_model]
  }
  function applyRotaryEmbedding(Q, K, sin, cos) {
    // Q, K shape: [batch, seq_len, nHeads, d_head]
    // We assume d_head is even for simplicity (rotary applies to pairs of dims).
    const seqLen = tf.shape(Q).arraySync()[1];
    const dHead = tf.shape(Q).arraySync()[3];
  
    // Split Q and K into even and odd parts (corresponding to sinusoidal pairs)
    const Q_even = Q.slice([0, 0, 0, 0], [-1, -1, -1, dHead/2]);
    const Q_odd  = Q.slice([0, 0, 0, dHead/2], [-1, -1, -1, dHead/2]);
    const K_even = K.slice([0, 0, 0, 0], [-1, -1, -1, dHead/2]);
    const K_odd  = K.slice([0, 0, 0, dHead/2], [-1, -1, -1, dHead/2]);
  
    // Expand sin, cos to [1, seq_len, 1, dHead/2] for broadcasting (if not already).
    const sinExp = tf.reshape(sin, [1, seqLen, 1, -1]);
    const cosExp = tf.reshape(cos, [1, seqLen, 1, -1]);
  
    // Apply rotation: 
    // Q_rotated_even = Q_even * cos - Q_odd * sin
    // Q_rotated_odd  = Q_even * sin + Q_odd * cos
    const Q_rot_even = Q_even.mul(cosExp).sub( Q_odd.mul(sinExp) );
    const Q_rot_odd  = Q_even.mul(sinExp).add( Q_odd.mul(cosExp) );
    // Similarly for K
    const K_rot_even = K_even.mul(cosExp).sub( K_odd.mul(sinExp) );
    const K_rot_odd  = K_even.mul(sinExp).add( K_odd.mul(cosExp) );
  
    // Concatenate the even and odd parts back together on the last dimension
    const Q_rot = tf.concat([Q_rot_even, Q_rot_odd], -1);
    const K_rot = tf.concat([K_rot_even, K_rot_odd], -1);
    return [Q_rot, K_rot];
  }
  // Weight matrices for FFN: W1 [d_model, d_ff], b1 [d_ff]; W2 [d_ff, d_model], b2 [d_model]
function feedForward(x, W1, b1, W2, b2) {
    // x shape: [batch, seq_len, d_model]
    let hidden = tf.matMul(x, W1).add(b1);    // linear projection to hidden dim (d_ff)
    // Activation (GELU or ReLU; we'll use GELU approximation via tf.erf for demonstration)
    hidden = tf.mul(0.5, tf.mul(hidden, tf.add(1, tf.erf(hidden.div(Math.sqrt(2))))));  // GELU formula
    let output = tf.matMul(hidden, W2).add(b2);  // project back to d_model
    return output;  // shape: [batch, seq_len, d_model]
  }
  function transformerBlock(x, attnWeights, ffnWeights, rotarySin, rotaryCos, nHeads) {
    // Unpack weights
    const {Wq, Wk, Wv, Wo} = attnWeights;
    const {W1, b1, W2, b2} = ffnWeights;
    const {gamma1, gamma2} = attnWeights;  // if using RMSNorm gamma for attention and FFN (pre-norm)
    
    // Pre-norm and Self-Attention
    const normed_x = rmsNorm(x, gamma1);
    const attn_out = multiHeadAttention(normed_x, Wq, Wk, Wv, Wo, nHeads, rotarySin, rotaryCos);
    const x_residual = x.add(attn_out);  // residual connection
  
    // Pre-norm and Feed-Forward
    const normed_attn = rmsNorm(x_residual, gamma2);
    const ffn_out = feedForward(normed_attn, W1, b1, W2, b2);
    const output = x_residual.add(ffn_out);  // second residual connection
  
    return output;
  }
  
  // To build a full N-layer Transformer (decoder-only):
  let x = inputEmbeddings;  // [batch, seq_len, d_model] from token embeddings
  for (let layer = 0; layer < N; layer++) {
    x = transformerBlock(x, attnWeights[layer], ffnWeights[layer], rotarySin, rotaryCos, nHeads);
  }
  const theta = 10000 ** (2 * (i/2) / d_head); 

