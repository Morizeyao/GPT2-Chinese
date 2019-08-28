DEFAULT_FILE_TYPE = ".json"

# 默认采用json文件作为输入的文件类型
# 如果自定义了文件类型，或者数据源，请return False
def is_default_file_type():
  return True

# 示例方法
# def load():
#     #根据文件编码类型，选择相应编码
#     with open("/user/npl/gpt/data/train.txt", 'r', encoding='gbk') as f:
#       print('reading lines')
#       #如果文件编码是GBK，才进行转换，否则lines = f.readlines()
#       lines = f.readlines();
#       lines = [line.replace('\n', ' [SEP] ') for line in lines]  # 用[SEP]表示换行, 段落之间使用SEP表示段落结束
#     return lines

# 请用相应的原编码加载文件
# 自定文件类型或者数据源必须实现此方法
# 最终返回列表，具体参考上面示例
def load():
 pass
