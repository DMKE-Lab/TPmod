import os
import pickle
import csv
import numpy as np

def load_data(inPath, fileName, fileName2=None):
    with open(os.path.join(inPath,fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head,rel,tail,time])
            times.add(time)
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()
    return np.asarray(quadrupleList), np.asarray(time)
    
def get_total_number(inPath,fileName):
    with open(os.path.join(inPath,fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])
        
train_data, train_times = load_data('./','train.txt')
valid_data, valid_times = load_data('./','valid.txt')
test_data, test_times = load_data('./','test.txt')

num_e, num_r = get_total_number('./','static.txt')
rel2val = {}
val = []
with open('rel2val.txt', 'r') as fr:
    for line in fr:
        line_split = line.split()
        rel2val[int(line_split[0])] = float(line_split[1])
for i in range(len(rel2val)):
    val.append(rel2val[i])
        
history_len = 10
time_len = 0
latest_t = 0
'''
final parameter:
    s_r_history
    r_o_history
    s_o_tendency
temp parameter:
    s_r_cache
    r_o_cache
    s_o_cache
    
    length
    value
'''

length = {}
length_o = {}
value = {}
value_o = {}

total_data = np.concatenate((train_data, valid_data, test_data), axis=0)
for train in total_data:
    s = train[0]
    r = train[1]
    o = train[2]
    if s not in value.keys():
        value[s] = {}
        length[s] = {}
    if o not in value[s].keys():
        value[s][o] = val[r]
        length[s][o] = 1
    else:
        value[s][o] += val[r]
        length[s][o] += 1
    if o not in value_o.keys():
        value_o[o] = {}
        length_o[o] = {}
    if s not in value_o[o].keys():
        value_o[o][s] = val[r]
        length_o[o][s] = 1
    else:
        value_o[o][s] += val[r]
        length_o[o][s] += 1

s_r_hist = [[]for _ in range(num_e)]
r_o_hist = [[]for _ in range(num_e)]
s_o_hist = [[]for _ in range(num_e)]
o_r_hist = [[]for _ in range(num_e)]
r_s_hist = [[]for _ in range(num_e)]
o_s_hist = [[]for _ in range(num_e)]

s_r_cache = [[]for _ in range(num_e)]
r_o_cache = [[[]for _ in range(num_r)]for _ in range(num_e)]
s_o_cache = [[[]for _ in range(num_r)]for _ in range(num_e)]
o_r_cache = [[]for _ in range(num_e)]
r_s_cache = [[[]for _ in range(num_r)]for _ in range(num_e)]
o_s_cache = [[[]for _ in range(num_r)]for _ in range(num_e)]

train_s_r_history = [[] for _ in range(len(train_data))]
train_r_o_history = [[] for _ in range(len(train_data))]
train_s_o_history = [[]for _ in range(len(train_data))]
train_o_r_history = [[] for _ in range(len(train_data))]
train_r_s_history = [[] for _ in range(len(train_data))]
train_o_s_history = [[]for _ in range(len(train_data))]

for i, train in enumerate(train_data):
    if i % 10000 == 0:
        print('train', i, len(train_data))
    t = train[3]
    if latest_t != t:
        for sub in range(num_e):
            if len(s_r_cache[sub]) != 0:
                if len(s_r_hist[sub]) >= history_len:
                    s_r_hist[sub].pop(0)
                    r_o_hist[sub].pop(0)
                    s_o_hist[sub].pop(0)
                s_r_hist[sub].append(s_r_cache[sub].copy())
                r_o = []
                s_o = []
                for rel in s_r_cache[sub]:
                    r_o.append(r_o_cache[sub][rel].copy())
                    o_val = []
                    for obj in r_o_cache[sub][rel]:
                        o_val.append(round(value[sub][obj]/length[sub][obj], 3))
                    s_o.append(o_val)
                    r_o_cache[sub][rel] = []
                r_o_hist[sub].append(r_o)
                s_o_hist[sub].append(s_o)
                s_r_cache[sub] = []
            if len(o_r_cache[sub]) != 0:
                if len(o_r_hist[sub]) >= history_len:
                    o_r_hist[sub].pop(0)
                    r_s_hist[sub].pop(0)
                    o_s_hist[sub].pop(0)
                o_r_hist[sub].append(o_r_cache[sub].copy())
                r_s = []
                o_s = []
                for rel in o_r_cache[sub]:
                    r_s.append(r_s_cache[sub][rel].copy())
                    s_val = []
                    for obj in r_s_cache[sub][rel]:
                        s_val.append(round(value_o[sub][obj]/length_o[sub][obj], 3))
                    o_s.append(s_val)
                    r_s_cache[sub][rel] = []
                r_s_hist[sub].append(r_s)
                o_s_hist[sub].append(o_s)
                o_r_cache[sub] = []
        latest_t = t
    
    s = train[0]
    r = train[1]
    o = train[2]
    
    train_s_r_history[i] = s_r_hist[s].copy()
    train_r_o_history[i] = r_o_hist[s].copy()
    train_s_o_history[i] = s_o_hist[s].copy()
    train_o_r_history[i] = o_r_hist[o].copy()
    train_r_s_history[i] = r_s_hist[o].copy()
    train_o_s_history[i] = o_s_hist[o].copy()
    if r not in s_r_cache[s]:
        s_r_cache[s].append(r)
    if o not in r_o_cache[s][r]:
        r_o_cache[s][r].append(o)
    
    if r not in o_r_cache[o]:
        o_r_cache[o].append(r)
    if s not in r_s_cache[o][r]:
        r_s_cache[o][r].append(s)

with open('train_s_r_history.txt', 'wb') as fp:
    pickle.dump(train_s_r_history, fp)
with open('train_r_o_history.txt', 'wb') as fp:
    pickle.dump(train_r_o_history, fp)
with open('train_s_o_history.txt', 'wb') as fp:
    pickle.dump(train_s_o_history, fp)
with open('train_o_r_history.txt', 'wb') as fp:
    pickle.dump(train_o_r_history, fp)
with open('train_r_s_history.txt', 'wb') as fp:
    pickle.dump(train_r_s_history, fp)
with open('train_o_s_history.txt', 'wb') as fp:
    pickle.dump(train_o_s_history, fp)

valid_s_r_history = [[] for _ in range(len(valid_data))]
valid_r_o_history = [[] for _ in range(len(valid_data))]
valid_s_o_history = [[] for _ in range(len(valid_data))]
valid_o_r_history = [[] for _ in range(len(valid_data))]
valid_r_s_history = [[] for _ in range(len(valid_data))]
valid_o_s_history = [[] for _ in range(len(valid_data))]

for i, valid in enumerate(valid_data):
    if i % 10000 == 0:
        print('train', i, len(valid_data))
    t = valid[3]
    if latest_t != t:
        for sub in range(num_e):
            if len(s_r_cache[sub]) != 0:
                if len(s_r_hist[sub]) >= history_len:
                    s_r_hist[sub].pop(0)
                    r_o_hist[sub].pop(0)
                    s_o_hist[sub].pop(0)
                s_r_hist[sub].append(s_r_cache[sub].copy())
                r_o = []
                s_o = []
                for rel in s_r_cache[sub]:
                    r_o.append(r_o_cache[sub][rel].copy())
                    o_val = []
                    for obj in r_o_cache[sub][rel]:
                        o_val.append(round(value[sub][obj] / length[sub][obj], 3))
                    s_o.append(o_val)
                    r_o_cache[sub][rel] = []
                r_o_hist[sub].append(r_o)
                s_o_hist[sub].append(s_o)
                s_r_cache[sub] = []
            if len(o_r_cache[sub]) != 0:
                if len(o_r_hist[sub]) >= history_len:
                    o_r_hist[sub].pop(0)
                    r_s_hist[sub].pop(0)
                    o_s_hist[sub].pop(0)
                o_r_hist[sub].append(o_r_cache[sub].copy())
                r_s = []
                o_s = []
                for rel in o_r_cache[sub]:
                    r_s.append(r_s_cache[sub][rel].copy())
                    s_val = []
                    for obj in r_s_cache[sub][rel]:
                        s_val.append(round(value_o[sub][obj] / length_o[sub][obj], 3))
                    o_s.append(s_val)
                    r_s_cache[sub][rel] = []
                r_s_hist[sub].append(r_s)
                o_s_hist[sub].append(o_s)
                o_r_cache[sub] = []
        latest_t = t
    
    s = valid[0]
    r = valid[1]
    o = valid[2]
    
    valid_s_r_history[i] = s_r_hist[s].copy()
    valid_r_o_history[i] = r_o_hist[s].copy()
    valid_s_o_history[i] = s_o_hist[s].copy()
    valid_o_r_history[i] = o_r_hist[o].copy()
    valid_r_s_history[i] = r_s_hist[o].copy()
    valid_o_s_history[i] = o_s_hist[o].copy()
    if r not in s_r_cache[s]:
        s_r_cache[s].append(r)
    if o not in r_o_cache[s][r]:
        r_o_cache[s][r].append(o)
    
    if r not in o_r_cache[o]:
        o_r_cache[o].append(r)
    if s not in r_s_cache[o][r]:
        r_s_cache[o][r].append(s)

with open('valid_s_r_history.txt', 'wb') as fp:
    pickle.dump(valid_s_r_history, fp)
with open('valid_r_o_history.txt', 'wb') as fp:
    pickle.dump(valid_r_o_history, fp)
with open('valid_s_o_history.txt', 'wb') as fp:
    pickle.dump(valid_s_o_history, fp)
with open('valid_o_r_history.txt', 'wb') as fp:
    pickle.dump(valid_o_r_history, fp)
with open('valid_r_s_history.txt', 'wb') as fp:
    pickle.dump(valid_r_s_history, fp)
with open('valid_o_s_history.txt', 'wb') as fp:
    pickle.dump(valid_o_s_history, fp)

test_s_r_history = [[] for _ in range(len(test_data))]
test_r_o_history = [[] for _ in range(len(test_data))]
test_s_o_history = [[] for _ in range(len(test_data))]
test_o_r_history = [[] for _ in range(len(test_data))]
test_r_s_history = [[] for _ in range(len(test_data))]
test_o_s_history = [[] for _ in range(len(test_data))]

for i, test in enumerate(test_data):
    if i % 10000 == 0:
        print('test', i, len(test_data))
    t = test[3]
    if latest_t != t:
        for sub in range(num_e):
            if len(s_r_cache[sub]) != 0:
                if len(s_r_hist[sub]) >= history_len:
                    s_r_hist[sub].pop(0)
                    r_o_hist[sub].pop(0)
                    s_o_hist[sub].pop(0)
                s_r_hist[sub].append(s_r_cache[sub].copy())
                r_o = []
                s_o = []
                for rel in s_r_cache[sub]:
                    r_o.append(r_o_cache[sub][rel].copy())
                    o_val = []
                    for obj in r_o_cache[sub][rel]:
                        o_val.append(round(value[sub][obj] / length[sub][obj], 3))
                    s_o.append(o_val)
                    r_o_cache[sub][rel] = []
                r_o_hist[sub].append(r_o)
                s_o_hist[sub].append(s_o)
                s_r_cache[sub] = []
            if len(o_r_cache[sub]) != 0:
                if len(o_r_hist[sub]) >= history_len:
                    o_r_hist[sub].pop(0)
                    r_s_hist[sub].pop(0)
                    o_s_hist[sub].pop(0)
                o_r_hist[sub].append(o_r_cache[sub].copy())
                r_s = []
                o_s = []
                for rel in o_r_cache[sub]:
                    r_s.append(r_s_cache[sub][rel].copy())
                    s_val = []
                    for obj in r_s_cache[sub][rel]:
                        s_val.append(round(value_o[sub][obj] / length_o[sub][obj], 3))
                    o_s.append(s_val)
                    r_s_cache[sub][rel] = []
                r_s_hist[sub].append(r_s)
                o_s_hist[sub].append(o_s)
                o_r_cache[sub] = []
        latest_t = t
    
    s = test[0]
    r = test[1]
    o = test[2]
    
    test_s_r_history[i] = s_r_hist[s].copy()
    test_r_o_history[i] = r_o_hist[s].copy()
    test_s_o_history[i] = s_o_hist[s].copy()
    test_o_r_history[i] = o_r_hist[o].copy()
    test_r_s_history[i] = r_s_hist[o].copy()
    test_o_s_history[i] = o_s_hist[o].copy()
    if r not in s_r_cache[s]:
        s_r_cache[s].append(r)
    if o not in r_o_cache[s][r]:
        r_o_cache[s][r].append(o)

    if r not in o_r_cache[o]:
        o_r_cache[o].append(r)
    if s not in r_s_cache[o][r]:
        r_s_cache[o][r].append(s)

with open('test_s_r_history.txt', 'wb') as fp:
    pickle.dump(test_s_r_history, fp)
with open('test_r_o_history.txt', 'wb') as fp:
    pickle.dump(test_r_o_history, fp)
with open('test_s_o_history.txt', 'wb') as fp:
    pickle.dump(test_s_o_history, fp)
with open('test_o_r_history.txt', 'wb') as fp:
    pickle.dump(test_o_r_history, fp)
with open('test_r_s_history.txt', 'wb') as fp:
    pickle.dump(test_r_s_history, fp)
with open('test_o_s_history.txt', 'wb') as fp:
    pickle.dump(test_o_s_history, fp)
