import sys
import pickle

if __name__ == '__main__':
    model = pickle.loads(sys.stdin.buffer.read())
    result = model.super_execute()
    sys.stdout.buffer.write(pickle.dumps(result))