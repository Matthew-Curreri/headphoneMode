class BPETokenizer {
  /**
   * Create a Byte Pair Encoding tokenizer.
   * @param {Object} modelData - JSON object with tokenizer model data.
   *   Expected format: { "vocab": {token: id, ...}, "merges": [ [tok1, tok2], ... ], "specialTokens": {token: id, ...} }.
   *   - "vocab": maps token strings (after byte-to-unicode encoding) to integer IDs.
   *   - "merges": list of pairs (each a two-element array or string pair) defining BPE merge rules in order.
   *   - "specialTokens": (optional) maps special token strings to reserved IDs.
   * @param {Object} [options={}] - Optional configuration.
   * @param {number} [options.maxConsecutiveChars=25000] - Max length of consecutive non-whitespace (or whitespace) sequence allowed.
   * @param {number} [options.maxChunkLength=400000] - Max length of text chunk to process at once (longer texts are chunked).
   */
  constructor (modelData, options = {}) {
    if (!modelData || !modelData.vocab || !modelData.merges) {
      throw new Error('Invalid model data: expected vocab and merges')
    }
    // Load vocabulary (token -> id)
    this.encoder = Object.assign({}, modelData.vocab)
    // Build decoder (id -> token)
    this.decoder = {}
    for (const [token, id] of Object.entries(this.encoder)) {
      this.decoder[id] = token
    }
    // Load special tokens if provided
    this.specialEncoder = {}
    this.specialDecoder = {}
    if (modelData.specialTokens) {
      for (const [st, id] of Object.entries(modelData.specialTokens)) {
        this.specialEncoder[st] = id
        this.specialDecoder[id] = st
      }
    }
    // Set limits for long sequences
    this.maxConsecutiveChars = options.maxConsecutiveChars ?? 25000
    this.maxChunkLength = options.maxChunkLength ?? 400000
    // Prepare BPE merge ranks dictionary (pair -> rank)
    // We use a delimiter character for pair keys that will not appear in tokens ('\x01' is a safe choice).
    this.bpeRanks = new Map()
    const merges = modelData.merges
    merges.forEach((pair, index) => {
      let [a, b] = Array.isArray(pair) ? pair : pair.split(/\s+/)
      // Key format "a\x01b"
      this.bpeRanks.set(`${a}\x01${b}`, index)
    })
    // Precompute the byte-to-unicode and unicode-to-byte maps for encoding bytes
    const { byteToUnicode, unicodeToByte } = BPETokenizer._initByteUnicodeMaps()
    this.byteToUnicode = byteToUnicode
    this.unicodeToByte = unicodeToByte
    // Compile the regex for pre-tokenization:
    // Pattern explanation:
    //   - Handles common contractions ('s, 't, 're, 've, 'm, 'll, 'd) case-insensitively.
    //   - Sequences of letters, sequences of digits, sequences of other symbols.
    //   - Leading space in those sequences is included if present.
    //   - Any remaining single whitespace character (to catch isolated spaces or newlines).
    const contraction = "'(?:(?:[sdmt]|ll|ve|re)|(?:[SDMT]|LL|VE|RE))" // include case variants
    const letterSeq = '[\\p{L}]+'
    const numberSeq = '[\\p{N}]+'
    const otherSeq = '[^\\p{L}\\p{N}\\s]+'
    const regexPattern = `${contraction}| ?${letterSeq}| ?${numberSeq}| ?${otherSeq}|\\s`
    this._tokenPattern = new RegExp(regexPattern, 'gu')
    // Cache for BPE results of already-processed words
    this.cache = new Map()
  }

  /**
   * Encode a text string into an array of token IDs.
   * Special tokens in the input (if defined in the model) will be extracted and encoded as single tokens.
   * @param {string} text - The input text to tokenize.
   * @returns {number[]} Array of token IDs.
   */
  encode (text) {
    if (text.length === 0) return []
    const tokens = []
    // If special tokens are defined, find them in the text and split around them
    if (Object.keys(this.specialEncoder).length > 0) {
      // Split text by occurrences of any special token
      // We'll identify the earliest occurrence of any special token in the remaining text iteratively.
      let i = 0
      while (i < text.length) {
        // Find the next special token occurrence
        let nextIndex = text.length
        let nextToken = null
        for (const st of Object.keys(this.specialEncoder)) {
          const idx = text.indexOf(st, i)
          if (idx !== -1 && idx < nextIndex) {
            nextIndex = idx
            nextToken = st
          }
        }
        if (nextToken === null || nextIndex === -1) {
          // No more special tokens in the remainder of the text
          const segment = text.slice(i)
          this._encodeNormalSegment(segment, tokens)
          break
        }
        // Encode any text before this special token
        const segment = text.slice(i, nextIndex)
        if (segment) {
          this._encodeNormalSegment(segment, tokens)
        }
        // Add the special token
        const tokenId = this.specialEncoder[nextToken]
        tokens.push(tokenId)
        // Advance index past the special token
        i = nextIndex + nextToken.length
      }
    } else {
      // No special tokens defined, encode the whole text normally
      this._encodeNormalSegment(text, tokens)
    }
    return tokens
  }

  /**
   * Decode an array of token IDs back into the original text string.
   * Special tokens (if any) are converted to their corresponding strings.
   * @param {number[]} tokenIds - Array of token IDs to decode.
   * @returns {string} The decoded text.
   */
  decode (tokenIds) {
    // Reconstruct the byte-sequence string from token IDs
    let text = ''
    for (const tokenId of tokenIds) {
      if (this.specialDecoder[tokenId] !== undefined) {
        // If it's a special token, append its string directly
        text += this.specialDecoder[tokenId]
      } else {
        const token = this.decoder[tokenId]
        if (token === undefined) {
          throw new Error(`Unknown token id: ${tokenId}`)
        }
        text += token
      }
    }
    // Map the unicode bytes back to actual bytes and decode to UTF-8
    const bytes = []
    for (let i = 0; i < text.length; i++) {
      const char = text[i]
      // Use the unicode-to-byte mapping (if char not in map, treat as char code)
      bytes.push(
        this.unicodeToByte[char] !== undefined
          ? this.unicodeToByte[char]
          : text.charCodeAt(i)
      )
    }
    // Convert byte array to string (UTF-8)
    // In modern environments, we can use TextDecoder:
    if (typeof TextDecoder !== 'undefined') {
      return new TextDecoder('utf-8').decode(new Uint8Array(bytes))
    } else {
      // Fallback for environments without TextDecoder
      let result = ''
      for (let j = 0; j < bytes.length; j++) {
        result += String.fromCharCode(bytes[j])
      }
      try {
        // decode URI component in case of multi-byte sequences
        return decodeURIComponent(escape(result))
      } catch {
        // If decoding fails, return raw result
        return result
      }
    }
  }

  /**
   * Internal method: encodes a normal text segment (with no special tokens).
   * This method applies the pre-tokenization regex, then BPE on each token.
   * It also handles splitting of long sequences to respect maxConsecutiveChars.
   * @param {string} segment - Text segment without special tokens.
   * @param {number[]} tokensOut - Array to append resulting token IDs to.
   * @private
   */
  _encodeNormalSegment (segment, tokensOut) {
    if (!segment) return
    // If the segment is very large, break it into smaller chunks
    // First, process in chunks of maxChunkLength to avoid excessive regex input size
    for (let start = 0; start < segment.length; start += this.maxChunkLength) {
      const chunk = segment.slice(start, start + this.maxChunkLength)
      // Further split chunk by long sequences of the same character class (whitespace vs non-whitespace)
      const subSegments = this._splitByLongestRun(chunk)
      for (const sub of subSegments) {
        if (sub === '') continue
        if (/^\s+$/.test(sub)) {
          // If the subsegment is purely whitespace, each whitespace char can be encoded directly
          // (Each whitespace like " " or "\n" should be in vocab as it's in byte range).
          // We can encode them one by one:
          for (const ch of sub) {
            const byte = ch.charCodeAt(0)
            // Map byte to token char via byteToUnicode (should map space/newline to a unicode char)
            const tokenChar = this.byteToUnicode[byte]
            const tokenId = this.encoder[tokenChar]
            if (tokenId !== undefined) {
              tokensOut.push(tokenId)
            } else {
              throw new Error(
                `Whitespace token '${ch}' not found in vocabulary.`
              )
            }
          }
          continue
        }
        // Use regex to find words/pieces in the sub-segment
        const matches = sub.matchAll(this._tokenPattern)
        for (const match of matches) {
          let token = match[0]
          // Convert token to bytes, then to the BPE-friendly unicode form
          const tokenBytes = BPETokenizer.textToUTF8Bytes(token)
          let tokenStr = ''
          for (const b of tokenBytes) {
            tokenStr += this.byteToUnicode[b]
          }
          // If we have seen this tokenStr in cache, reuse its encoded form
          if (this.cache.has(tokenStr)) {
            tokensOut.push(...this.cache.get(tokenStr))
            continue
          }
          // Perform BPE merging on the token string
          const tokenIds = this._bpeMerge(tokenStr)
          // Cache the result for this tokenStr
          this.cache.set(tokenStr, tokenIds)
          // Append to output list
          tokensOut.push(...tokenIds)
        }
      }
    }
  }

  /**
   * Internal helper: Apply BPE merges to a token string (already in byte-unicode form).
   * Returns an array of token IDs corresponding to the final merged tokens.
   * @param {string} tokenStr - The token as a string of byte-mapped unicode characters.
   * @returns {number[]} token IDs after BPE merging.
   * @private
   */
  _bpeMerge (tokenStr) {
    // If the token is a single character, return its ID directly (no merge possible)
    if (tokenStr.length <= 1) {
      return this.encoder[tokenStr] !== undefined
        ? [this.encoder[tokenStr]]
        : []
    }
    // Represent the token as an array of symbols (each symbol is a string, initially one char each)
    let symbols = [...tokenStr] // spread splits the string into array of single-char strings
    // Greedily merge pairs until no mergeable pair remains
    while (symbols.length > 1) {
      // Find the adjacent pair with the lowest BPE rank
      let minPairKey = null
      let minRank = Infinity
      let minIndex = -1
      for (let i = 0; i < symbols.length - 1; i++) {
        const pairKey = `${symbols[i]}\x01${symbols[i + 1]}`
        const rank = this.bpeRanks.get(pairKey)
        if (rank !== undefined && rank < minRank) {
          minRank = rank
          minPairKey = pairKey
          minIndex = i
        }
      }
      // If no adjacent pair is mergeable (not in bpeRanks), we're done
      if (minPairKey === null) break
      // Merge the best pair found: replace symbols[i] and symbols[i+1] with their concatenation
      const first = symbols[minIndex]
      const second = symbols[minIndex + 1]
      symbols.splice(minIndex, 2, first + second)
    }
    // At this point, each symbol in the array is a final token (in byte-unicode form).
    // Map each symbol to its token ID using the vocabulary (encoder).
    return symbols.map(sym => {
      if (!(sym in this.encoder)) {
        throw new Error(
          `Token '${sym}' not found in vocabulary during BPE merge.`
        )
      }
      return this.encoder[sym]
    })
  }

  /**
   * Internal helper: Split a text chunk into sub-segments such that no sub-segment has a run
   * of consecutive whitespace or non-whitespace characters longer than the maxConsecutiveChars limit.
   * @param {string} text - The text chunk to split.
   * @returns {string[]} Array of sub-segment strings.
   * @private
   */
  _splitByLongestRun (text) {
    if (this.maxConsecutiveChars <= 0) return [text]
    const result = []
    let lastBreak = 0
    let lastCharType = null // "ws" for whitespace, "non" for non-whitespace
    let runLength = 0
    for (let i = 0; i < text.length; i++) {
      const isWhitespace = /\s/.test(text[i])
      const charType = isWhitespace ? 'ws' : 'non'
      if (charType === lastCharType) {
        // continuing the same run
        runLength++
      } else {
        // run type changed
        runLength = 1
        lastCharType = charType
      }
      if (runLength >= this.maxConsecutiveChars) {
        // We have reached the max allowed length of a run. Break here.
        result.push(text.slice(lastBreak, i + 1)) // include this char in the current segment
        lastBreak = i + 1
        runLength = 0
        lastCharType = null
      }
    }
    // Push the remaining segment if any
    if (lastBreak < text.length) {
      result.push(text.slice(lastBreak))
    }
    return result
  }

  /**
   * Convert a text string to an array of UTF-8 bytes.
   * @param {string} text - Input text.
   * @returns {number[]} Array of byte values (0-255).
   * @static
   */
  static textToUTF8Bytes (text) {
    // Use TextEncoder if available for efficiency
    if (typeof TextEncoder !== 'undefined') {
      return Array.from(new TextEncoder().encode(text))
    }
    // Fallback: encode using escape/unescape (may not handle all cases in older environments)
    const utf8 = unescape(encodeURIComponent(text))
    const bytes = []
    for (let i = 0; i < utf8.length; i++) {
      bytes.push(utf8.charCodeAt(i))
    }
    return bytes
  }

  /**
   * Initialize byte-to-unicode and unicode-to-byte lookup tables for the byte fallback encoding.
   * This implements the same logic as OpenAI's bytes_to_unicode() in Python.
   * @returns {Object} An object { byteToUnicode: Object, unicodeToByte: Object }.
   * @static
   */
  static _initByteUnicodeMaps () {
    // Create list of byte values that will directly map to characters
    const bs = []
    // printable ASCII from '!' (33) to '~' (126)
    for (let i = 33; i <= 126; i++) bs.push(i)
    // extended characters from '¡' (161) to '¬' (172)
    for (let i = 161; i <= 172; i++) bs.push(i)
    // extended characters from '®' (174) to 'ÿ' (255)
    for (let i = 174; i <= 255; i++) bs.push(i)
    const cs = bs.slice() // copy initial set
    // Add mappings for bytes not in the initial list
    let n = 0
    for (let b = 0; b < 256; b++) {
      if (!bs.includes(b)) {
        bs.push(b)
        cs.push(256 + n)
        n++
      }
    }
    // bs and cs are now parallel lists of equal length
    const byteToUnicode = {}
    const unicodeToByte = {}
    for (let i = 0; i < bs.length; i++) {
      const byteVal = bs[i]
      const uniVal = cs[i]
      const ch = String.fromCharCode(uniVal)
      byteToUnicode[byteVal] = ch
      unicodeToByte[ch] = byteVal
    }
    return { byteToUnicode, unicodeToByte }
  }
}
// Assume modelData is the JSON object containing vocab, merges, special tokens.
const tokenizer = new BPETokenizer(modelData)

const text = 'Hello, world!'
const tokenIds = tokenizer.encode(text)
console.log(tokenIds)

const decoded = tokenizer.decode(tokenIds)
console.log(decoded) // "Hello, world!"
