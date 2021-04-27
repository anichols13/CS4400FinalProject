import pandas as pd
import numpy as np, os
from os.path import join
from sklearn.ensemble import RandomForestClassifier
import Levenshtein as lev
import py_entitymatching as em

ltable= em.read_csv_metadata("ltable.csv", key="id")
rtable= em.read_csv_metadata("rtable.csv", key="id")
train= pd.read_csv("train.csv")



ob= em.OverlapBlocker()
new=ob.block_tables(ltable, rtable, 'brand', 'brand',
	l_output_attrs = ['id','title', 'category', 'brand','modelno','price'],
    r_output_attrs = ['id','title', 'category', 'brand','modelno','price'],
    overlap_size =1, show_progress=False)


def pairs(ltable, rtable, cs):
    ltable.index = ltable.id
    rtable.index = rtable.id
    pairs = np.array(cs)
    ltable_tuples = ltable.loc[pairs[:, 0], :]
    rtable_tuples = rtable.loc[pairs[:, 1], :]
    ltable_tuples.columns = [c + "_l" for c in ltable_tuples.columns]
    rtable_tuples.columns = [c + "_r" for c in rtable_tuples.columns]
    ltable_tuples.reset_index(inplace=True, drop=True)
    rtable_tuples.reset_index(inplace=True, drop=True)
    p = pd.concat([ltable_tuples, rtable_tuples], axis=1)
    return p


td = pairs(ltable, rtable, list(map(tuple, train[["ltable_id", "rtable_id"]].values)))
td['label'] = train['label']
td['_id'] = td.index
c = td.columns.tolist()
c = c[-1:] + c[:-1]
td = td[c]
tl = train.label.values


em.set_key(td, '_id')
em.set_ltable(td, ltable)
em.set_fk_ltable(td, 'id_l')

em.set_rtable(td, rtable)
em.set_fk_rtable(td, 'id_r')



IJ = em.split_train_test(td, train_proportion = 0.9)
I = IJ['train']
J = IJ['test']


rf = RandomForestClassifier(class_weight="balanced", random_state=0)

token = em.get_tokenizers_for_matching()
simulation = em.get_sim_funs_for_matching()
r_att_type = em.get_attr_types(rtable)
r_att_type['title'] = 'str_bt_5w_10w'
bk = em.get_attr_corres(ltable, rtable)
bk['corres'].pop(0)
bk['corres'].pop(1)
bk['corres'].pop(1)
bk['corres'].pop(1)
bk['corres'].pop(1) #leaving just title


feature_table = em.get_features(ltable, rtable, em.get_attr_types(ltable),
    r_att_type, bk, token, simulation)

H = em.extract_feature_vecs(I, feature_table=feature_table)
Hnew=H.iloc[:,3:]
H['label'] = train['label']
list_store=Hnew.values.tolist()

L = em.extract_feature_vecs(new, feature_table=feature_table)

L_new=L.iloc[:,3:]
list_store_2=L_new.values.tolist()


rf.fit(list_store,tl[:4500])

predictions = rf.predict(list_store_2)




matching_pairs = new.loc[predictions == 1, ["ltable_id", "rtable_id"]]
matching_pairs = list(map(tuple, matching_pairs.values))

matching_pairs_in_training = td.loc[tl == 1, ["id_l", "id_r"]]
matching_pairs_in_training = set(list(map(tuple, matching_pairs_in_training.values)))

pred_pairs = [pair for pair in matching_pairs if
              pair not in matching_pairs_in_training]  
pred_pairs = np.array(pred_pairs)
pred_df = pd.DataFrame(pred_pairs, columns=["ltable_id", "rtable_id"])
pred_df.to_csv("output.csv", index=False)



