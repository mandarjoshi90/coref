import json
import sys
import util

def print_clusters(data_file):
    f = open(data_file)
    for i, line in enumerate(f):
      data = json.loads(line)
      text = util.flatten(data['sentences'])
      # clusters = [[text[s:e+1] for s,e in cluster] for cluster in data['clusters']]
      #print(text)
      for ci, cluster in enumerate(data['clusters']):
        spans = [text[s:e+1] for s,e in cluster]
        print(i, ci, spans)
      if i > 5:
        break

if __name__ == '__main__':
    data_file = sys.argv[1]
    print_clusters(data_file)
