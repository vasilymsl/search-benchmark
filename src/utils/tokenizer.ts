const STOP_WORDS = new Set([
  'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
  'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
  'could', 'should', 'may', 'might', 'shall', 'can', 'need', 'dare',
  'ought', 'used', 'it', 'its', 'this', 'that', 'these', 'those',
  'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours', 'you', 'your',
  'yours', 'he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them',
  'their', 'theirs', 'what', 'which', 'who', 'whom', 'whose',
  'where', 'when', 'how', 'not', 'no', 'nor', 'as', 'if', 'then',
  'than', 'too', 'very', 'just', 'about', 'above', 'after', 'again',
  'all', 'also', 'am', 'any', 'because', 'before', 'between', 'both',
  'each', 'few', 'get', 'got', 'here', 'into', 'more', 'most', 'new',
  'now', 'only', 'other', 'over', 'own', 'same', 'so', 'some', 'still',
  'such', 'take', 'through', 'under', 'up', 'while',
]);

/**
 * Porter Stemmer (simplified).
 * Handles common English suffixes.
 */
function porterStem(word: string): string {
  if (word.length < 3) return word;

  // Step 1a
  if (word.endsWith('sses')) word = word.slice(0, -2);
  else if (word.endsWith('ies')) word = word.slice(0, -2);
  else if (!word.endsWith('ss') && word.endsWith('s')) word = word.slice(0, -1);

  // Step 1b
  if (word.endsWith('eed')) {
    if (word.length > 4) word = word.slice(0, -1);
  } else if (word.endsWith('ed') && word.length > 4) {
    word = word.slice(0, -2);
    if (word.endsWith('at') || word.endsWith('bl') || word.endsWith('iz')) {
      word += 'e';
    }
  } else if (word.endsWith('ing') && word.length > 5) {
    word = word.slice(0, -3);
    if (word.endsWith('at') || word.endsWith('bl') || word.endsWith('iz')) {
      word += 'e';
    }
  }

  // Step 1c
  if (word.endsWith('y') && word.length > 2) {
    word = word.slice(0, -1) + 'i';
  }

  // Step 2 (simplified)
  const step2: Record<string, string> = {
    ational: 'ate', tional: 'tion', enci: 'ence', anci: 'ance',
    izer: 'ize', alli: 'al', entli: 'ent', eli: 'e',
    ousli: 'ous', ization: 'ize', ation: 'ate', ator: 'ate',
    alism: 'al', iveness: 'ive', fulness: 'ful', ousness: 'ous',
    aliti: 'al', iviti: 'ive', biliti: 'ble',
  };
  for (const [suffix, replacement] of Object.entries(step2)) {
    if (word.endsWith(suffix) && word.length - suffix.length > 1) {
      word = word.slice(0, -suffix.length) + replacement;
      break;
    }
  }

  // Step 3 (simplified)
  const step3: Record<string, string> = {
    icate: 'ic', ative: '', alize: 'al', iciti: 'ic',
    ical: 'ic', ful: '', ness: '',
  };
  for (const [suffix, replacement] of Object.entries(step3)) {
    if (word.endsWith(suffix) && word.length - suffix.length > 1) {
      word = word.slice(0, -suffix.length) + replacement;
      break;
    }
  }

  return word;
}

export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter((t) => t.length > 1 && !STOP_WORDS.has(t))
    .map(porterStem);
}

export function tokenizeRaw(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter((t) => t.length > 0);
}
