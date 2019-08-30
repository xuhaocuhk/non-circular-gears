if __name__ == '__main__':
    import os

    model_formatstr = """
- center_point: [0.7, 0.7]
  k: 1
  name: {0}
  sample_num: 1024
  smooth: 32
  tooth_height: 0.03
  tooth_num: 32"""

    dir = os.listdir(r'C:\Projects\gears\silhouette\new')
    with open('output.txt', 'w') as out:
        for file in dir:
            if '.txt' in file:
                assert file[:-4] + '.jpg' in dir or file[:-4] + '.png' in dir
                print(model_formatstr.format(file[:-4]), file=out)

    # new models: {'wandou', 'bat', 'qingtianwa', 'tree', 'pistol', 'drop', 'mohaima', 'chinese_cup', 'wingsuit', 'gun', 'heart', 'china_map', 'sweden_map', 'lvmaochong', 'australia', 'woniu', 'shark', 'shoes', 'trump2', 'guo', 'key', 'liyuwang', 'chicken_leg', 'turtle', 'pikachu2', 'koala', 'kangaroo', 'usmap', 'fighter', 'kabi'}
