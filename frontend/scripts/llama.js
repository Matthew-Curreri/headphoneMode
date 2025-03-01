class ModelArgs {
    constructor() {
        this.dim = 4096;
        this.n_layers = 32;
        this.n_heads = 32;
        this.n_kv_heads = null;
        this.vocab_size = -1; // Defined later by tokenizer
        this.multiple_of = 256; // Ensure SwiGLU hidden layer size is a multiple of 2
        this.ffn_dim_multiplier = null;
        this.norm_eps = 1e-5;
        this.max_batch_size = 32;
        this.max_seq_len = 2048;
    }
}

class RMSNorm {
    constructor(dim, eps = 1e-6) {
        this.eps = eps;
        this.weight = new Float32Array(dim).fill(1);
    }

    _norm(x) {
        let meanSquares = x.map(val => val ** 2).reduce((sum, val) => sum + val, 0) / x.length;
        let invRms = 1 / Math.sqrt(meanSquares + this.eps);
        return x.map(val => val * invRms);
    }

    forward(x) {
        let output = this._norm(x);
        return output.map((val, i) => val * this.weight[i]);
    }
}

function precomputeFreqsCis(dim, end, theta = 10000.0) {
    let freqs = new Float32Array(dim / 2).map((_, i) => 1.0 / (theta ** (i / dim)));
    let t = new Float32Array(end).map((_, i) => i);
    let freqsCis = t.map(val => Math.exp(1j * val * freqs)); // Simulating complex numbers
    return freqsCis;
}

function reshapeForBroadcast(freqsCis, x) {
    let shape = x.map((_, i) => (i === 1 || i === x.length - 1 ? x[i] : 1));
    return freqsCis.reshape(shape);
}

function applyRotaryEmb(xq, xk, freqsCis) {
    let xq_out = xq.map((val, i) => val * freqsCis[i]);
    let xk_out = xk.map((val, i) => val * freqsCis[i]);
    return [xq_out, xk_out];
}

function repeatKV(x, n_rep) {
    if (n_rep === 1) return x;
    return Array(n_rep).fill(x).flat();
}

class Attention {
    constructor(args) {
        this.n_kv_heads = args.n_kv_heads || args.n_heads;
        this.n_local_heads = args.n_heads;
        this.n_local_kv_heads = this.n_kv_heads;
        this.n_rep = this.n_local_heads / this.n_local_kv_heads;
        this.head_dim = args.dim / args.n_heads;
        this.wq = new Float32Array(args.dim * this.head_dim);
        this.wk = new Float32Array(args.dim * this.head_dim);
        this.wv = new Float32Array(args.dim * this.head_dim);
        this.wo = new Float32Array(args.dim * this.head_dim);
    }

    forward(x, start_pos, freqsCis, mask) {
        let xq = x.map(val => val * this.wq);
        let xk = x.map(val => val * this.wk);
        let xv = x.map(val => val * this.wv);
        [xq, xk] = applyRotaryEmb(xq, xk, freqsCis);
        let keys = repeatKV(xk, this.n_rep);
        let values = repeatKV(xv, this.n_rep);
        let scores = xq.map((val, i) => val * keys[i]);
        if (mask) scores = scores.map((val, i) => val + mask[i]);
        let softmaxed = scores.map(val => Math.exp(val) / scores.reduce((sum, v) => sum + Math.exp(v), 0));
        let output = softmaxed.map((val, i) => val * values[i]);
        return output.map(val => val * this.wo);
    }
}

class Transformer {
    constructor(params) {
        this.params = params;
        this.vocab_size = params.vocab_size;
        this.n_layers = params.n_layers;
        this.norm = new RMSNorm(params.dim, params.norm_eps);
        this.layers = Array.from({ length: params.n_layers }, () => new Attention(params));
    }

    forward(tokens, start_pos) {
        let h = tokens;
        let freqsCis = precomputeFreqsCis(this.params.dim / this.params.n_heads, this.params.max_seq_len * 2);
        let mask = new Float32Array(tokens.length).fill(-Infinity);
        for (let i = 0; i < tokens.length; i++) {
            mask[i] = 0;
        }
        for (let layer of this.layers) {
            h = layer.forward(h, start_pos, freqsCis, mask);
        }
        return this.norm.forward(h);
    }
}
