import numpy as np

class DependencyData:
   '''
    class for handling dependency trees for questions and answers
    take in a list of dependency trees, topologically sort them,
    and for each tree return a list of word indices, relation indices, parent node indices, and an answer index
   '''

   def __init__(self):
	self.vocab = {}
	self.relations = {}
	self.answers = {}
	self.train_answers = {}

   def scan_vocab(self, corpus, train_index=0):
       ret = []
       for datasetIndex,data in enumerate(corpus.datasets):
            ret.append(self.transform(data, True))
       return ret

   def sort_datum(self, datum):
        size = len(datum)
        Adj = {}
        # build the graph
        for i in range (1, size):
            parent = datum[i][2]
            if parent not in Adj:
                Adj[parent] = []
            Adj[parent].append( i )

        result = []
        queue = [ 0 ]
        cnt = 1
        nextLeve = 0
        # bfs search
        while len(queue) > 0:
            temp_array = []
            nextLevel = 0
            for _ in range( cnt ):
                temp = queue.pop(0)
                temp_array.append( temp )
                if temp in Adj:
                    for x in Adj[temp]:
                        nextLevel += 1
                        queue.append( x )
            temp_array.sort()
            for x in temp_array:
                result.append( x )

            cnt = nextLevel

        return result
        raise NotImplementedError

   def match_embeddings(self, model):
      '''
      takes in a word2vec model and return a matrix of embeddings
      
      model - gensim Word2Vec object
      '''

      embeddings = [None for i in range(len(self.vocab))]

      for word in self.vocab:
         if word in model:
            embedding = model[word]
         else:
            embedding = np.random.uniform(-.2, .2, model.layer1_size)
         embeddings[self.vocab[word]] = embedding

      return np.array(embeddings)

   def stop_indices(self, stop_words):
      '''given a list of stop words, return their indices'''
      
      stop_indices = set()
      for word in stop_words:
         if word in self.vocab:
            stop_indices.add(self.vocab[word])
      return stop_indices
   
   def transform(self, data, initialize=False):
      '''return the indices for the words, relations, parents, and answer'''
      output = []
      for datum,answer in data:
          #do topological sort here, also remove nodes that don't have parents (including ROOT)
          #order the nodes left to right with the root as the rightmost node
          indices = self.sort_datum(datum)
          idxs, rel_idxs, p = [], [], []
          parentLookup = {}
          for index in indices:
              if datum[index][2] is None:
                  parentLookup[index] = len(indices)-1
                  continue
              if index not in parentLookup:
                  parentLookup[index] = len(indices)-len(parentLookup)-1

              word, relation, parent = datum[index]

              if initialize:
                 if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
                 if relation not in self.relations:
                    self.relations[relation] = len(self.relations)

              idxs.insert(0, self.vocab[word])
              rel_idxs.insert(0, self.relations[relation])
              p.insert(0, parentLookup[parent])

          if initialize:
             if answer not in self.answers:
                 if answer not in self.vocab:
                     self.vocab[answer] = len(self.vocab)
                 self.answers[answer] = self.vocab[answer]

          output.append((idxs, rel_idxs, p, self.answers[answer]))

      return output
