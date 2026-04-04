export interface Document {
  id: string;
  title: string;
  text: string;
}

export interface Query {
  id: string;
  text: string;
}

export interface Qrels {
  [queryId: string]: { [docId: string]: number };
}

export interface Dataset {
  corpus: Document[];
  queries: Query[];
  qrels: Qrels;
  /** doc.id -> index in corpus array */
  docIdToIndex: Map<string, number>;
}
