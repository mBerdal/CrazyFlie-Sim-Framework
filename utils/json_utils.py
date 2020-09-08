import json

def read_json(filename):
  with open(filename) as json_file:
      return json.load(json_file)