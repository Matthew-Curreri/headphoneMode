const fs = require('fs');

class Tokenizer {
    constructor(modelPath) {
        if (!fs.existsSync(modelPath)) {
            throw new Error(`Model file not found: ${modelPath}`);
        }
        this.model = this.loadModel(modelPath);
        this.nWords = this.model.length;
        this.bosId = this.nWords; // Simulating BOS ID
        this.eosId = this.nWords + 1; // Simulating EOS ID
        this.padId = this.nWords + 2; // Simulating PAD ID
    }

    loadModel(modelPath) {
        const data = fs.readFileSync(modelPath, 'utf-8');
        return data.split('\n').map(line => line.trim()).filter(line => line);
    }

    bpeEncode(text) {
        const words = text.split(/\s+/);
        let tokens = [];
        for (let word of words) {
            let subwords = this.bytePairEncoding(word);
            tokens.push(...subwords);
        }
        return tokens;
    }

    bytePairEncoding(word) {
        let subwords = [];
        let maxLen = Math.min(word.length, 4); // Limit to 4-char subwords for simplicity
        for (let i = 0; i < word.length; i += maxLen) {
            let subword = word.substring(i, i + maxLen);
            let idx = this.model.indexOf(subword);
            subwords.push(idx !== -1 ? idx : this.padId);
        }
        return subwords;
    }

    encode(text, bos = false, eos = false) {
        if (typeof text !== 'string') {
            throw new TypeError('Input must be a string');
        }
        let tokens = this.bpeEncode(text);
        if (bos) tokens.unshift(this.bosId);
        if (eos) tokens.push(this.eosId);
        return tokens;
    }

    decode(tokenIds) {
        if (!Array.isArray(tokenIds)) {
            throw new TypeError('Input must be an array of token IDs');
        }
        return tokenIds.map(id => (id >= 0 && id < this.nWords ? this.model[id] : '')).join(' ');
    }
}

module.exports = Tokenizer;
