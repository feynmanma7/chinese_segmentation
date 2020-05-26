from src.dict_utils import load_dict

if __name__ == '__main__':
    user_dict_path = '../../data/people.dict.pkl'
    user_dict = load_dict(dict_path=user_dict_path)
    print('Len user_dict=%d' % len(user_dict))
