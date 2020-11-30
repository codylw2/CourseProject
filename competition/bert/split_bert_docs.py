import os
import json
import time


def load_json(json_path):
    if os.path.isfile(json_path):
        with open(json_path, 'r') as f_json:
            j_dict = json.load(f_json)
            return j_dict
    else:
        raise Exception('json file does not exist: {0}'.format(json_path))


def write_json(j_dict, f_name, script_dir):
    with open(os.path.join(script_dir, f_name), 'w') as json_f:
        json.dump(j_dict, json_f)
    print('json file written: {0}'.format(os.path.join(script_dir, f_name)))


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def drop_keys(docs):
    for key, val in docs.items():
        val.pop('text', None)
        docs[key] = val
    return docs


if __name__ == '__main__':
    t_start = time.time()

    script_dir = os.path.dirname(__file__)
    docs = load_json(os.path.join('..', 'test_docs.json'))
    uid_keys = list(docs['uid'].keys())
    del(docs['id'])
    for key in uid_keys:
        del(docs['uid'][key]['title'])
        del(docs['uid'][key]['text'])
        del(docs['uid'][key]['abstract'])
        del(docs['uid'][key]['pmcid'])
        del(docs['uid'][key]['pmc_json_files'])
        del(docs['uid'][key]['pdf_json_files'])

    print(docs['uid'][uid_keys[0]].get('text', 'no text key'))

    splits = 10
    chunk_size = int(len(uid_keys) / splits)
    print('chunk size: {}'.format(chunk_size))

    uid_list = list()
    for idx, i in enumerate(range(0, len(docs['uid'].keys()), chunk_size)):
        if idx == splits:
            uid_list = uid_keys[i:]
            print('{}:'.format(i))
        else:
            uid_list = uid_keys[i:i+chunk_size]
            print('{}:{}'.format(i, i+chunk_size))

        print('creating new dict {}'.format(idx))
        new_dict = {'uid': dict()}
        for uid in uid_list:
            new_dict['uid'][uid] = docs['uid'][uid]
            # del(docs['uid'][uid])

        write_json(new_dict, 'json_data', 'test_docs_{0}.json'.format(str(idx).zfill(2)), script_dir)
        del new_dict

    # expected run time 7824.042961835861 seconds
    print('script ran in {} seconds'.format(time.time()-t_start))