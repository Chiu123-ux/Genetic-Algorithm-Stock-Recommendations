import pickle

def preview_pickle_file(file_path, num_elements=5):
    try:
        with open(file_path, 'rb') as f:
            # 读取第一个对象
            obj = pickle.load(f)
            
            # 根据对象类型，查看前几项
            if isinstance(obj, (list, tuple)):
                print("前几项内容（列表/元组）：")
                for i, item in enumerate(obj[:num_elements]):
                    print(f"  [{i}]: {item}")
            elif isinstance(obj, dict):
                print("前几项内容（字典）：")
                for i, (key, value) in enumerate(obj.items()):
                    if i >= num_elements:
                        break
                    print(f"  Key: {key}, Value: {value}")
            elif hasattr(obj, '__repr__'):
                print("对象内容（repr 表示）：")
                print(repr(obj))
            else:
                print("对象内容（原始）：")
                print(obj)
    except Exception as e:
        print(f"读取文件时出错: {e}")

preview_pickle_file('data\\factor_data.pickle')