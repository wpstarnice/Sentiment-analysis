from creat_data import baidu, ali, tencent, config
import pymysql
import pandas as pd
import numpy as np
import os

DT = config.DT
conn = pymysql.connect(**DT, charset='utf8')
cur = conn.cursor()

start_n, length = 0, 5000
new_query = 'select `id`,`evaluation` from `t_ebiz_comment_1407714269270694_78091_10782_1` limit %d,%d' % (
    start_n, length)
cur.execute(query=new_query)
data = list(cur.fetchall())
data = pd.DataFrame(data, columns=['id', 'evaluation'])

texts = data['evaluation']

results_baidu = baidu.creat_label(texts, interface='API')
# results_baidu_0 = pd.DataFrame(results_baidu, columns=['evaluation',
#                                                        'label',
#                                                        'confidence',
#                                                        'positive_prob',
#                                                        'negative_prob',
#                                                        'ret',
#                                                        'msg'])
# results_baidu_0['label'] = np.where(results_baidu_0['label'] == 2,
#                                     '正面',
#                                     np.where(results_baidu_0['label'] == 1, '中性', '负面'))

results_ali = ali.creat_label(texts)
# results_ali_0 = pd.DataFrame(results_ali, columns=['evaluation',
#                                                    'label',
#                                                    'ret',
#                                                    'msg'])
# results_ali_0['label'] = np.where(results_ali_0['label'] == '1', '正面',
#                                   np.where(results_ali_0['label'] == '0', '中性',
#                                            np.where(results_ali_0['label'] == '-1', '负面', '非法')))

results_tencent = tencent.creat_label(texts)
# results_tencent_0 = pd.DataFrame(results_tencent, columns=['evaluation',
#                                                            'label',
#                                                            'confidence',
#                                                            'ret',
#                                                            'msg'])
# results_tencent_0['label'] = np.where(results_tencent_0['label'] == 1, '正面',
#                                       np.where(results_tencent_0['label'] == 0, '中性', '负面'))

results_all = [[texts[i],
                results_baidu[i][1], results_baidu[i][6],
                results_ali[i][1], results_ali[i][3],
                results_tencent[i][1], results_tencent[i][4]] for i in range(len(texts))]
results_dataframe = pd.DataFrame(results_all,
                                 columns=['evaluation',
                                          'label_baidu', 'msg_baidu',
                                          'label_ali', 'msg_ali',
                                          'label_tencent', 'msg_tencent'])
results_dataframe['label_baidu'] = np.where(results_dataframe['label_baidu'] == 2,
                                            '正面',
                                            np.where(results_dataframe['label_baidu'] == 1, '中性', '负面'))
results_dataframe['label_ali'] = np.where(results_dataframe['label_ali'] == '1', '正面',
                                  np.where(results_dataframe['label_ali'] == '0', '中性',
                                           np.where(results_dataframe['label_ali'] == '-1', '负面', '非法')))
results_dataframe['label_tencent'] = np.where(results_dataframe['label_tencent'] == 1, '正面',
                                      np.where(results_dataframe['label_tencent'] == 0, '中性', '负面'))
results_dataframe.to_excel(config.label_path +
                           '/t_ebiz_comment_1407714269270694_78091_10782_1/%d_%d.xlsx' % (start_n, start_n + length),
                           index=False)
