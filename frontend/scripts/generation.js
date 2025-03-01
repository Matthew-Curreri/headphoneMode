class Llama {
    static async build(ckptDir, tokenizerPath, maxSeqLen, maxBatchSize) {
        const model = await Llama.loadModel(ckptDir);
        const tokenizer = new Tokenizer(tokenizerPath);
        model.vocabSize = tokenizer.nWords;
        return new Llama(model, tokenizer, maxSeqLen, maxBatchSize);
    }

    constructor(model, tokenizer, maxSeqLen, maxBatchSize) {
        this.model = model;
        this.tokenizer = tokenizer;
        this.maxSeqLen = maxSeqLen;
        this.maxBatchSize = maxBatchSize;
    }

    async generate(promptTokens, maxGenLen, temperature = 0.6, topP = 0.9) {
        let totalLen = Math.min(this.maxSeqLen, maxGenLen + Math.max(...promptTokens.map(t => t.length)));
        let tokens = new Array(this.maxBatchSize).fill().map(() => new Array(totalLen).fill(this.tokenizer.padId));

        for (let i = 0; i < promptTokens.length; i++) {
            tokens[i].splice(0, promptTokens[i].length, ...promptTokens[i]);
        }

        for (let curPos = Math.min(...promptTokens.map(t => t.length)); curPos < totalLen; curPos++) {
            let logits = await this.model.forward(tokens.map(t => t.slice(0, curPos)));
            let nextTokens = this.sampleTopP(logits.map(l => l[l.length - 1]), topP, temperature);
            for (let i = 0; i < tokens.length; i++) {
                tokens[i][curPos] = nextTokens[i];
            }
        }
        return tokens;
    }

    sampleTopP(logits, p, temperature) {
        let sampled = logits.map(logit => {
            let expLogits = logit.map(l => Math.exp(l / temperature));
            let sumExp = expLogits.reduce((a, b) => a + b, 0);
            let probs = expLogits.map(e => e / sumExp);
            let sortedIndices = [...probs.keys()].sort((a, b) => probs[b] - probs[a]);
            let cumulativeProb = 0;
            let topIndices = sortedIndices.filter(i => {
                cumulativeProb += probs[i];
                return cumulativeProb <= p;
            });
            let renormalizedProbs = topIndices.map(i => probs[i] / cumulativeProb);
            let rand = Math.random();
            let chosenIndex = topIndices.findIndex((_, i) => rand <= renormalizedProbs.slice(0, i + 1).reduce((a, b) => a + b, 0));
            return topIndices[chosenIndex];
        });
        return sampled;
    }
}

class Tokenizer {
    constructor(modelData) {
        this.model = this.loadModel(modelData);
        this.nWords = this.model.length;
        this.bosId = this.nWords;
        this.eosId = this.nWords + 1;
        this.padId = this.nWords + 2;
    }

    loadModel(modelData) {
        return modelData.split('\n').map(line => line.trim()).filter(line => line);
    }

    async encode(text, bos = false, eos = false) {
        let tokens = text.split(/\s+/).map(word => this.model.indexOf(word)).map(idx => (idx !== -1 ? idx : this.padId));
        if (bos) tokens.unshift(this.bosId);
        if (eos) tokens.push(this.eosId);
        return tokens;
    }

    decode(tokenIds) {
        return tokenIds.map(id => (id >= 0 && id < this.nWords ? this.model[id] : '')).join(' ');
    }
}
