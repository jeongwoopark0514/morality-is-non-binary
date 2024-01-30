#!/usr/bin/env python
# coding: utf-8

# In[1]:

#  MFTC Moral Similarity Table - Both Train and Test all together
import pandas as pd
import numpy as np
from simcse import SimCSE


# In[2]:


simcse_model = SimCSE("model/sup-batch32/large-lr5e-05-ep2-seq64-batch32-temp0.1")


# In[4]:


# Read in single labels file for train set and test set.
mftc_train_set = pd.read_csv("data/single labels all.csv")
# mftc_test_set = pd.read_csv("data/single_labels_test.csv", index_col=0)


# In[9]:


# First process train set
mftc_train_care_df = mftc_train_set[mftc_train_set["care"] == 1]
mftc_train_harm_df = mftc_train_set[mftc_train_set["harm"] == 1]
mftc_train_fairness_df = mftc_train_set[mftc_train_set["fairness"] == 1]
mftc_train_cheating_df = mftc_train_set[mftc_train_set["cheating"] == 1]
mftc_train_loyalty_df = mftc_train_set[mftc_train_set["loyalty"] == 1]
mftc_train_betrayal_df = mftc_train_set[mftc_train_set["betrayal"] == 1]
mftc_train_authority_df = mftc_train_set[mftc_train_set["authority"] == 1]
mftc_train_subversion_df = mftc_train_set[mftc_train_set["subversion"] == 1]
mftc_train_purity_df = mftc_train_set[mftc_train_set["purity"] == 1]
mftc_train_degradation_df = mftc_train_set[mftc_train_set["degradation"] == 1]
mftc_train_non_moral_df = mftc_train_set[mftc_train_set["non-moral"] == 1]
#
#
# # In[10]:
#
#
# # First process test set
# mftc_test_care_df = mftc_test_set[mftc_test_set["care"] == 1]
# mftc_test_harm_df = mftc_test_set[mftc_test_set["harm"] == 1]
# mftc_test_fairness_df = mftc_test_set[mftc_test_set["fairness"] == 1]
# mftc_test_cheating_df = mftc_test_set[mftc_test_set["cheating"] == 1]
# mftc_test_loyalty_df = mftc_test_set[mftc_test_set["loyalty"] == 1]
# mftc_test_betrayal_df = mftc_test_set[mftc_test_set["betrayal"] == 1]
# mftc_test_authority_df = mftc_test_set[mftc_test_set["authority"] == 1]
# mftc_test_subversion_df = mftc_test_set[mftc_test_set["subversion"] == 1]
# mftc_test_purity_df = mftc_test_set[mftc_test_set["purity"] == 1]
# mftc_test_degradation_df = mftc_test_set[mftc_test_set["degradation"] == 1]
# mftc_test_non_moral_df = mftc_test_set[mftc_test_set["non-moral"] == 1]
#



# MERGE
# mftc_all_care_df = pd.concat([mftc_train_care_df, mftc_test_care_df], ignore_index=True)
# mftc_all_harm_df = pd.concat([mftc_train_harm_df, mftc_test_harm_df], ignore_index=True)
# mftc_all_fairness_df = pd.concat([mftc_train_fairness_df, mftc_test_fairness_df], ignore_index=True)
# mftc_all_cheating_df = pd.concat([mftc_train_cheating_df, mftc_test_cheating_df], ignore_index=True)
# mftc_all_loyalty_df = pd.concat([mftc_train_loyalty_df, mftc_test_loyalty_df], ignore_index=True)
# mftc_all_betrayal_df = pd.concat([mftc_train_betrayal_df, mftc_test_betrayal_df], ignore_index=True)
# mftc_all_authority_df = pd.concat([mftc_train_authority_df, mftc_test_authority_df], ignore_index=True)
# mftc_all_subversion_df = pd.concat([mftc_train_subversion_df, mftc_test_subversion_df], ignore_index=True)
# mftc_all_purity_df = pd.concat([mftc_train_purity_df, mftc_test_purity_df], ignore_index=True)
# mftc_all_degradation_df = pd.concat([mftc_train_degradation_df, mftc_test_degradation_df], ignore_index=True)
# mftc_all_non_moral_df = pd.concat([mftc_train_non_moral_df, mftc_test_non_moral_df], ignore_index=True)
#
# merged_single_labels = pd.concat([mftc_all_care_df, mftc_all_harm_df, mftc_all_fairness_df, mftc_all_cheating_df, mftc_all_loyalty_df, mftc_all_betrayal_df, mftc_all_authority_df, mftc_all_subversion_df, mftc_all_purity_df, mftc_all_degradation_df, mftc_all_non_moral_df], ignore_index=True)
# merged_single_labels.to_csv("single labels all.csv", index=False)
#
# # In[ ]:
#
#
# # Care
care_harm_sim = simcse_model.similarity(mftc_train_care_df["processed"].tolist(), mftc_train_harm_df["processed"].tolist())
care_cheating_sim = simcse_model.similarity(mftc_train_care_df["processed"].tolist(), mftc_train_cheating_df["processed"].tolist())
care_betrayal_sim = simcse_model.similarity(mftc_train_care_df["processed"].tolist(), mftc_train_betrayal_df["processed"].tolist())
care_subversion_sim = simcse_model.similarity(mftc_train_care_df["processed"].tolist(), mftc_train_subversion_df["processed"].tolist())
care_degradation_sim = simcse_model.similarity(mftc_train_care_df["processed"].tolist(), mftc_train_degradation_df["processed"].tolist())

care_care_sim = simcse_model.similarity(mftc_train_care_df["processed"].tolist(), mftc_train_care_df["processed"].tolist())
care_fairness_sim = simcse_model.similarity(mftc_train_care_df["processed"].tolist(), mftc_train_fairness_df["processed"].tolist())
care_loyalty_sim = simcse_model.similarity(mftc_train_care_df["processed"].tolist(), mftc_train_loyalty_df["processed"].tolist())
care_authority_sim = simcse_model.similarity(mftc_train_care_df["processed"].tolist(), mftc_train_authority_df["processed"].tolist())
care_purity_sim = simcse_model.similarity(mftc_train_care_df["processed"].tolist(), mftc_train_purity_df["processed"].tolist())

care_non_moral_sim = simcse_model.similarity(mftc_train_care_df["processed"].tolist(), mftc_train_non_moral_df["processed"].tolist())


print("care - care: ", care_care_sim.mean())
print("care - fairness: ", care_fairness_sim.mean())
print("care - loyalty: ", care_loyalty_sim.mean())
print("care - authority: ", care_authority_sim.mean())
print("care - purity: ", care_purity_sim.mean())

print("care - harm: ", care_harm_sim.mean())
print("care - cheating: ", care_cheating_sim.mean())
print("care - betrayal: ", care_betrayal_sim.mean())
print("care - subversion: ", care_subversion_sim.mean())
print("care - degradation: ", care_degradation_sim.mean())

print("care - non-moral: ", care_non_moral_sim.mean())

care_vec_train = [care_care_sim.mean(), care_fairness_sim.mean(), care_loyalty_sim.mean(), care_authority_sim.mean(), care_purity_sim.mean(),
            care_harm_sim.mean(), care_cheating_sim.mean(), care_betrayal_sim.mean(), care_subversion_sim.mean(), care_degradation_sim.mean(), care_non_moral_sim.mean()]


# In[ ]:


# save care numpy
np.save("morality_table_mftc/care_mftc_all.npy", care_vec_train)


# In[ ]:


# fairness
fairness_harm_sim = simcse_model.similarity(mftc_train_fairness_df["processed"].tolist(), mftc_train_harm_df["processed"].tolist())
fairness_cheating_sim = simcse_model.similarity(mftc_train_fairness_df["processed"].tolist(), mftc_train_cheating_df["processed"].tolist())
fairness_betrayal_sim = simcse_model.similarity(mftc_train_fairness_df["processed"].tolist(), mftc_train_betrayal_df["processed"].tolist())
fairness_subversion_sim = simcse_model.similarity(mftc_train_fairness_df["processed"].tolist(), mftc_train_subversion_df["processed"].tolist())
fairness_degradation_sim = simcse_model.similarity(mftc_train_fairness_df["processed"].tolist(), mftc_train_degradation_df["processed"].tolist())

fairness_care_sim = simcse_model.similarity(mftc_train_fairness_df["processed"].tolist(), mftc_train_care_df["processed"].tolist())
fairness_fairness_sim = simcse_model.similarity(mftc_train_fairness_df["processed"].tolist(), mftc_train_fairness_df["processed"].tolist())
fairness_loyalty_sim = simcse_model.similarity(mftc_train_fairness_df["processed"].tolist(), mftc_train_loyalty_df["processed"].tolist())
fairness_authority_sim = simcse_model.similarity(mftc_train_fairness_df["processed"].tolist(), mftc_train_authority_df["processed"].tolist())
fairness_purity_sim = simcse_model.similarity(mftc_train_fairness_df["processed"].tolist(), mftc_train_purity_df["processed"].tolist())

fairness_non_moral_sim = simcse_model.similarity(mftc_train_fairness_df["processed"].tolist(), mftc_train_non_moral_df["processed"].tolist())


print("fairness - care: ", fairness_care_sim.mean())
print("fairness - fairness: ", fairness_fairness_sim.mean())
print("fairness - loyalty: ", fairness_loyalty_sim.mean())
print("fairness - authority: ", fairness_authority_sim.mean())
print("fairness - purity: ", fairness_purity_sim.mean())

print("fairness - harm: ", fairness_harm_sim.mean())
print("fairness - cheating: ", fairness_cheating_sim.mean())
print("fairness - betrayal: ", fairness_betrayal_sim.mean())
print("fairness - subversion: ", fairness_subversion_sim.mean())
print("fairness - degradation: ", fairness_degradation_sim.mean())

print("fairness - non-moral: ", fairness_non_moral_sim.mean())

fairness_vec_train = [fairness_care_sim.mean(), fairness_fairness_sim.mean(), fairness_loyalty_sim.mean(), fairness_authority_sim.mean(), fairness_purity_sim.mean(),
               fairness_harm_sim.mean(), fairness_cheating_sim.mean(), fairness_betrayal_sim.mean(), fairness_subversion_sim.mean(), fairness_degradation_sim.mean(), fairness_non_moral_sim.mean()]

np.save("morality_table_mftc/fairness_mftc_all.npy", fairness_vec_train)


# In[ ]:



# loyalty
loyalty_harm_sim = simcse_model.similarity(mftc_train_loyalty_df["processed"].tolist(), mftc_train_harm_df["processed"].tolist())
loyalty_cheating_sim = simcse_model.similarity(mftc_train_loyalty_df["processed"].tolist(), mftc_train_cheating_df["processed"].tolist())
loyalty_betrayal_sim = simcse_model.similarity(mftc_train_loyalty_df["processed"].tolist(), mftc_train_betrayal_df["processed"].tolist())
loyalty_subversion_sim = simcse_model.similarity(mftc_train_loyalty_df["processed"].tolist(), mftc_train_subversion_df["processed"].tolist())
loyalty_degradation_sim = simcse_model.similarity(mftc_train_loyalty_df["processed"].tolist(), mftc_train_degradation_df["processed"].tolist())

loyalty_care_sim = simcse_model.similarity(mftc_train_loyalty_df["processed"].tolist(), mftc_train_care_df["processed"].tolist())
loyalty_fairness_sim = simcse_model.similarity(mftc_train_loyalty_df["processed"].tolist(), mftc_train_fairness_df["processed"].tolist())
loyalty_loyalty_sim = simcse_model.similarity(mftc_train_loyalty_df["processed"].tolist(), mftc_train_loyalty_df["processed"].tolist())
loyalty_authority_sim = simcse_model.similarity(mftc_train_loyalty_df["processed"].tolist(), mftc_train_authority_df["processed"].tolist())
loyalty_purity_sim = simcse_model.similarity(mftc_train_loyalty_df["processed"].tolist(), mftc_train_purity_df["processed"].tolist())

loyalty_non_moral_sim = simcse_model.similarity(mftc_train_loyalty_df["processed"].tolist(), mftc_train_non_moral_df["processed"].tolist())



print("loyalty - care: ", loyalty_care_sim.mean())
print("loyalty - fairness: ", loyalty_fairness_sim.mean())
print("loyalty - loyalty: ", loyalty_loyalty_sim.mean())
print("loyalty - authority: ", loyalty_authority_sim.mean())
print("loyalty - purity: ", loyalty_purity_sim.mean())

print("loyalty - harm: ", loyalty_harm_sim.mean())
print("loyalty - cheating: ", loyalty_cheating_sim.mean())
print("loyalty - betrayal: ", loyalty_betrayal_sim.mean())
print("loyalty - subversion: ", loyalty_subversion_sim.mean())
print("loyalty - degradation: ", loyalty_degradation_sim.mean())

print("loyalty - non-moral: ", loyalty_non_moral_sim.mean())

loyalty_vec_train = [loyalty_care_sim.mean(), loyalty_fairness_sim.mean(), loyalty_loyalty_sim.mean(), loyalty_authority_sim.mean(), loyalty_purity_sim.mean(),
              loyalty_harm_sim.mean(), loyalty_cheating_sim.mean(), loyalty_betrayal_sim.mean(), loyalty_subversion_sim.mean().mean(), loyalty_degradation_sim.mean(), loyalty_non_moral_sim.mean()]

np.save("morality_table_mftc/loyalty_mftc_all.npy", loyalty_vec_train)


# In[ ]:


# authority
authority_harm_sim = simcse_model.similarity(mftc_train_authority_df["processed"].tolist(), mftc_train_harm_df["processed"].tolist())
authority_cheating_sim = simcse_model.similarity(mftc_train_authority_df["processed"].tolist(), mftc_train_cheating_df["processed"].tolist())
authority_betrayal_sim = simcse_model.similarity(mftc_train_authority_df["processed"].tolist(), mftc_train_betrayal_df["processed"].tolist())
authority_subversion_sim = simcse_model.similarity(mftc_train_authority_df["processed"].tolist(), mftc_train_subversion_df["processed"].tolist())
authority_degradation_sim = simcse_model.similarity(mftc_train_authority_df["processed"].tolist(), mftc_train_degradation_df["processed"].tolist())

authority_care_sim = simcse_model.similarity(mftc_train_authority_df["processed"].tolist(), mftc_train_care_df["processed"].tolist())
authority_fairness_sim = simcse_model.similarity(mftc_train_authority_df["processed"].tolist(), mftc_train_fairness_df["processed"].tolist())
authority_loyalty_sim = simcse_model.similarity(mftc_train_authority_df["processed"].tolist(), mftc_train_loyalty_df["processed"].tolist())
authority_authority_sim = simcse_model.similarity(mftc_train_authority_df["processed"].tolist(), mftc_train_authority_df["processed"].tolist())
authority_purity_sim = simcse_model.similarity(mftc_train_authority_df["processed"].tolist(), mftc_train_purity_df["processed"].tolist())

authority_non_moral_sim = simcse_model.similarity(mftc_train_authority_df["processed"].tolist(), mftc_train_non_moral_df["processed"].tolist())




print("authority - care: ", authority_care_sim.mean())
print("authority - fairness: ", authority_fairness_sim.mean())
print("authority - loyalty: ", authority_loyalty_sim.mean())
print("authority - authority: ", authority_authority_sim.mean())
print("authority - purity: ", authority_purity_sim.mean())

print("authority - harm: ", authority_harm_sim.mean())
print("authority - cheating: ", authority_cheating_sim.mean())
print("authority - betrayal: ", authority_betrayal_sim.mean())
print("authority - subversion: ", authority_subversion_sim.mean())
print("authority - degradation: ", authority_degradation_sim.mean())

print("authority - non-moral: ", authority_non_moral_sim.mean())

authority_vec_train = [authority_care_sim.mean(), authority_fairness_sim.mean(), authority_loyalty_sim.mean(), authority_authority_sim.mean(), authority_purity_sim.mean(),
                authority_harm_sim.mean(), authority_cheating_sim.mean(), authority_betrayal_sim.mean(), authority_subversion_sim.mean(), authority_degradation_sim.mean(), authority_non_moral_sim.mean()]

np.save("morality_table_mftc/authority_mftc_all.npy", authority_vec_train)


# In[ ]:


# purity
purity_harm_sim = simcse_model.similarity(mftc_train_purity_df["processed"].tolist(), mftc_train_harm_df["processed"].tolist())
purity_cheating_sim = simcse_model.similarity(mftc_train_purity_df["processed"].tolist(), mftc_train_cheating_df["processed"].tolist())
purity_betrayal_sim = simcse_model.similarity(mftc_train_purity_df["processed"].tolist(), mftc_train_betrayal_df["processed"].tolist())
purity_subversion_sim = simcse_model.similarity(mftc_train_purity_df["processed"].tolist(), mftc_train_subversion_df["processed"].tolist())
purity_degradation_sim = simcse_model.similarity(mftc_train_purity_df["processed"].tolist(), mftc_train_degradation_df["processed"].tolist())

purity_care_sim = simcse_model.similarity(mftc_train_purity_df["processed"].tolist(), mftc_train_care_df["processed"].tolist())
purity_fairness_sim = simcse_model.similarity(mftc_train_purity_df["processed"].tolist(), mftc_train_fairness_df["processed"].tolist())
purity_loyalty_sim = simcse_model.similarity(mftc_train_purity_df["processed"].tolist(), mftc_train_loyalty_df["processed"].tolist())
purity_authority_sim = simcse_model.similarity(mftc_train_purity_df["processed"].tolist(), mftc_train_authority_df["processed"].tolist())
purity_purity_sim = simcse_model.similarity(mftc_train_purity_df["processed"].tolist(), mftc_train_purity_df["processed"].tolist())

purity_non_moral_sim = simcse_model.similarity(mftc_train_purity_df["processed"].tolist(), mftc_train_non_moral_df["processed"].tolist())



print("purity - care: ", purity_care_sim.mean())
print("purity - fairness: ", purity_fairness_sim.mean())
print("purity - loyalty: ", purity_loyalty_sim.mean())
print("purity - authority: ", purity_authority_sim.mean())
print("purity - purity: ", purity_purity_sim.mean())

print("purity - harm: ", purity_harm_sim.mean())
print("purity - cheating: ", purity_cheating_sim.mean())
print("purity - betrayal: ", purity_betrayal_sim.mean())
print("purity - subversion: ", purity_subversion_sim.mean())
print("purity - degradation: ", purity_degradation_sim.mean())

print("purity - non-moral: ", purity_non_moral_sim.mean())

purity_vec_train = [purity_care_sim.mean(), purity_fairness_sim.mean(), purity_loyalty_sim.mean(), purity_authority_sim.mean(), purity_purity_sim.mean(),
             purity_harm_sim.mean(), purity_cheating_sim.mean(), purity_betrayal_sim.mean(), purity_subversion_sim.mean(), purity_degradation_sim.mean(), purity_non_moral_sim.mean()]

np.save("morality_table_mftc/purity_mftc_all.npy", purity_vec_train)


# In[ ]:


# Harm

harm_harm_sim = simcse_model.similarity(mftc_train_harm_df["processed"].tolist(), mftc_train_harm_df["processed"].tolist())
harm_cheating_sim = simcse_model.similarity(mftc_train_harm_df["processed"].tolist(), mftc_train_cheating_df["processed"].tolist())
harm_betrayal_sim = simcse_model.similarity(mftc_train_harm_df["processed"].tolist(), mftc_train_betrayal_df["processed"].tolist())
harm_subversion_sim = simcse_model.similarity(mftc_train_harm_df["processed"].tolist(), mftc_train_subversion_df["processed"].tolist())
harm_degradation_sim = simcse_model.similarity(mftc_train_harm_df["processed"].tolist(), mftc_train_degradation_df["processed"].tolist())

harm_care_sim = simcse_model.similarity(mftc_train_harm_df["processed"].tolist(), mftc_train_care_df["processed"].tolist())
harm_fairness_sim = simcse_model.similarity(mftc_train_harm_df["processed"].tolist(), mftc_train_fairness_df["processed"].tolist())
harm_loyalty_sim = simcse_model.similarity(mftc_train_harm_df["processed"].tolist(), mftc_train_loyalty_df["processed"].tolist())
harm_authority_sim = simcse_model.similarity(mftc_train_harm_df["processed"].tolist(), mftc_train_authority_df["processed"].tolist())
harm_purity_sim = simcse_model.similarity(mftc_train_harm_df["processed"].tolist(), mftc_train_purity_df["processed"].tolist())

harm_non_moral_sim = simcse_model.similarity(mftc_train_harm_df["processed"].tolist(), mftc_train_non_moral_df["processed"].tolist())



print("harm - care: ", harm_care_sim.mean())
print("harm - fairness: ", harm_fairness_sim.mean())
print("harm - loyalty: ", harm_loyalty_sim.mean())
print("harm - authority: ", harm_authority_sim.mean())
print("harm - purity: ", harm_purity_sim.mean())

print("harm - harm: ", harm_harm_sim.mean())
print("harm - cheating: ", harm_cheating_sim.mean())
print("harm - betrayal: ", harm_betrayal_sim.mean())
print("harm - subversion: ", harm_subversion_sim.mean())
print("harm - degradation: ", harm_degradation_sim.mean())

print("harm - non-moral: ", harm_non_moral_sim.mean())

harm_vec_train = [harm_care_sim.mean(), harm_fairness_sim.mean(), harm_loyalty_sim.mean(), harm_authority_sim.mean(), harm_purity_sim.mean(),
            harm_harm_sim.mean(), harm_cheating_sim.mean(), harm_betrayal_sim.mean().mean(), harm_subversion_sim.mean(), harm_degradation_sim.mean()]

np.save("morality_table_mftc/harm_mftc_all.npy", harm_vec_train)


# In[ ]:


# Harm

harm_harm_sim = simcse_model.similarity(mftc_train_harm_df["processed"].tolist(), mftc_train_harm_df["processed"].tolist())
harm_cheating_sim = simcse_model.similarity(mftc_train_harm_df["processed"].tolist(), mftc_train_cheating_df["processed"].tolist())
harm_betrayal_sim = simcse_model.similarity(mftc_train_harm_df["processed"].tolist(), mftc_train_betrayal_df["processed"].tolist())
harm_subversion_sim = simcse_model.similarity(mftc_train_harm_df["processed"].tolist(), mftc_train_subversion_df["processed"].tolist())
harm_degradation_sim = simcse_model.similarity(mftc_train_harm_df["processed"].tolist(), mftc_train_degradation_df["processed"].tolist())

harm_care_sim = simcse_model.similarity(mftc_train_harm_df["processed"].tolist(), mftc_train_care_df["processed"].tolist())
harm_fairness_sim = simcse_model.similarity(mftc_train_harm_df["processed"].tolist(), mftc_train_fairness_df["processed"].tolist())
harm_loyalty_sim = simcse_model.similarity(mftc_train_harm_df["processed"].tolist(), mftc_train_loyalty_df["processed"].tolist())
harm_authority_sim = simcse_model.similarity(mftc_train_harm_df["processed"].tolist(), mftc_train_authority_df["processed"].tolist())
harm_purity_sim = simcse_model.similarity(mftc_train_harm_df["processed"].tolist(), mftc_train_purity_df["processed"].tolist())

harm_non_moral_sim = simcse_model.similarity(mftc_train_harm_df["processed"].tolist(), mftc_train_non_moral_df["processed"].tolist())



print("harm - care: ", harm_care_sim.mean())
print("harm - fairness: ", harm_fairness_sim.mean())
print("harm - loyalty: ", harm_loyalty_sim.mean())
print("harm - authority: ", harm_authority_sim.mean())
print("harm - purity: ", harm_purity_sim.mean())

print("harm - harm: ", harm_harm_sim.mean())
print("harm - cheating: ", harm_cheating_sim.mean())
print("harm - betrayal: ", harm_betrayal_sim.mean())
print("harm - subversion: ", harm_subversion_sim.mean())
print("harm - degradation: ", harm_degradation_sim.mean())

print("harm - non-moral: ", harm_non_moral_sim.mean())

harm_vec_train = [harm_care_sim.mean(), harm_fairness_sim.mean(), harm_loyalty_sim.mean(), harm_authority_sim.mean(), harm_purity_sim.mean(),
            harm_harm_sim.mean(), harm_cheating_sim.mean(), harm_betrayal_sim.mean().mean(), harm_subversion_sim.mean(), harm_degradation_sim.mean(), harm_non_moral_sim.mean()]

np.save("morality_table_mftc/harm_mftc_all.npy", harm_vec_train)


# In[ ]:


# Cheating

cheating_harm_sim = simcse_model.similarity(mftc_train_cheating_df["processed"].tolist(), mftc_train_harm_df["processed"].tolist())
cheating_cheating_sim = simcse_model.similarity(mftc_train_cheating_df["processed"].tolist(), mftc_train_cheating_df["processed"].tolist())
cheating_betrayal_sim = simcse_model.similarity(mftc_train_cheating_df["processed"].tolist(), mftc_train_betrayal_df["processed"].tolist())
cheating_subversion_sim = simcse_model.similarity(mftc_train_cheating_df["processed"].tolist(), mftc_train_subversion_df["processed"].tolist())
cheating_degradation_sim = simcse_model.similarity(mftc_train_cheating_df["processed"].tolist(), mftc_train_degradation_df["processed"].tolist())

cheating_care_sim = simcse_model.similarity(mftc_train_cheating_df["processed"].tolist(), mftc_train_care_df["processed"].tolist())
cheating_fairness_sim = simcse_model.similarity(mftc_train_cheating_df["processed"].tolist(), mftc_train_fairness_df["processed"].tolist())
cheating_loyalty_sim = simcse_model.similarity(mftc_train_cheating_df["processed"].tolist(), mftc_train_loyalty_df["processed"].tolist())
cheating_authority_sim = simcse_model.similarity(mftc_train_cheating_df["processed"].tolist(), mftc_train_authority_df["processed"].tolist())
cheating_purity_sim = simcse_model.similarity(mftc_train_cheating_df["processed"].tolist(), mftc_train_purity_df["processed"].tolist())

cheating_non_moral_sim = simcse_model.similarity(mftc_train_cheating_df["processed"].tolist(), mftc_train_non_moral_df["processed"].tolist())


print("cheating - care: ", cheating_care_sim.mean())
print("cheating - fairness: ", cheating_fairness_sim.mean())
print("cheating - loyalty: ", cheating_loyalty_sim.mean())
print("cheating - authority: ", cheating_authority_sim.mean())
print("cheating - purity: ", cheating_purity_sim.mean())

print("cheating - harm: ", cheating_harm_sim.mean())
print("cheating - cheating: ", cheating_cheating_sim.mean())
print("cheating - betrayal: ", cheating_betrayal_sim.mean())
print("cheating - subversion: ", cheating_subversion_sim.mean())
print("cheating - degradation: ", cheating_degradation_sim.mean())

print("cheating - non-moral: ", cheating_non_moral_sim.mean())

cheating_vec_train = [cheating_care_sim.mean(), cheating_fairness_sim.mean(), cheating_loyalty_sim.mean().mean(), cheating_authority_sim.mean(), cheating_purity_sim.mean(),
               cheating_harm_sim.mean(), cheating_cheating_sim.mean(), cheating_betrayal_sim.mean(), cheating_subversion_sim.mean(), cheating_degradation_sim.mean(), cheating_non_moral_sim.mean()]

np.save("morality_table_mftc/cheating_mftc_all.npy", cheating_vec_train)


# In[ ]:


# Betrayal

betrayal_harm_sim = simcse_model.similarity(mftc_train_betrayal_df["processed"].tolist(), mftc_train_harm_df["processed"].tolist())
betrayal_cheating_sim = simcse_model.similarity(mftc_train_betrayal_df["processed"].tolist(), mftc_train_cheating_df["processed"].tolist())
betrayal_betrayal_sim = simcse_model.similarity(mftc_train_betrayal_df["processed"].tolist(), mftc_train_betrayal_df["processed"].tolist())
betrayal_subversion_sim = simcse_model.similarity(mftc_train_betrayal_df["processed"].tolist(), mftc_train_subversion_df["processed"].tolist())
betrayal_degradation_sim = simcse_model.similarity(mftc_train_betrayal_df["processed"].tolist(), mftc_train_degradation_df["processed"].tolist())

betrayal_care_sim = simcse_model.similarity(mftc_train_betrayal_df["processed"].tolist(), mftc_train_care_df["processed"].tolist())
betrayal_fairness_sim = simcse_model.similarity(mftc_train_betrayal_df["processed"].tolist(), mftc_train_fairness_df["processed"].tolist())
betrayal_loyalty_sim = simcse_model.similarity(mftc_train_betrayal_df["processed"].tolist(), mftc_train_loyalty_df["processed"].tolist())
betrayal_authority_sim = simcse_model.similarity(mftc_train_betrayal_df["processed"].tolist(), mftc_train_authority_df["processed"].tolist())
betrayal_purity_sim = simcse_model.similarity(mftc_train_betrayal_df["processed"].tolist(), mftc_train_purity_df["processed"].tolist())

betrayal_non_moral_sim = simcse_model.similarity(mftc_train_betrayal_df["processed"].tolist(), mftc_train_non_moral_df["processed"].tolist())


print("betrayal - care: ", betrayal_care_sim.mean())
print("betrayal - fairness: ", betrayal_fairness_sim.mean())
print("betrayal - loyalty: ", betrayal_loyalty_sim.mean())
print("betrayal - authority: ", betrayal_authority_sim.mean())
print("betrayal - purity: ", betrayal_purity_sim.mean())

print("betrayal - harm: ", betrayal_harm_sim.mean())
print("betrayal - cheating: ", betrayal_cheating_sim.mean())
print("betrayal - betrayal: ", betrayal_betrayal_sim.mean())
print("betrayal - subversion: ", betrayal_subversion_sim.mean())
print("betrayal - degradation: ", betrayal_degradation_sim.mean())

print("betrayal - non-moral: ", betrayal_non_moral_sim.mean())


betrayal_vec_train = [betrayal_care_sim.mean(), betrayal_fairness_sim.mean(), betrayal_loyalty_sim.mean(), betrayal_authority_sim.mean(), betrayal_purity_sim.mean(),
               betrayal_harm_sim.mean(), betrayal_cheating_sim.mean(), betrayal_betrayal_sim.mean(), betrayal_subversion_sim.mean(), betrayal_degradation_sim.mean(), betrayal_non_moral_sim.mean()]

np.save("morality_table_mftc/betrayal_mftc_all.npy", betrayal_vec_train)


# In[ ]:


# subversion

subversion_harm_sim = simcse_model.similarity(mftc_train_subversion_df["processed"].tolist(), mftc_train_harm_df["processed"].tolist())
subversion_cheating_sim = simcse_model.similarity(mftc_train_subversion_df["processed"].tolist(), mftc_train_cheating_df["processed"].tolist())
subversion_betrayal_sim = simcse_model.similarity(mftc_train_subversion_df["processed"].tolist(), mftc_train_betrayal_df["processed"].tolist())
subversion_subversion_sim = simcse_model.similarity(mftc_train_subversion_df["processed"].tolist(), mftc_train_subversion_df["processed"].tolist())
subversion_degradation_sim = simcse_model.similarity(mftc_train_subversion_df["processed"].tolist(), mftc_train_degradation_df["processed"].tolist())

subversion_care_sim = simcse_model.similarity(mftc_train_subversion_df["processed"].tolist(), mftc_train_care_df["processed"].tolist())
subversion_fairness_sim = simcse_model.similarity(mftc_train_subversion_df["processed"].tolist(), mftc_train_fairness_df["processed"].tolist())
subversion_loyalty_sim = simcse_model.similarity(mftc_train_subversion_df["processed"].tolist(), mftc_train_loyalty_df["processed"].tolist())
subversion_authority_sim = simcse_model.similarity(mftc_train_subversion_df["processed"].tolist(), mftc_train_authority_df["processed"].tolist())
subversion_purity_sim = simcse_model.similarity(mftc_train_subversion_df["processed"].tolist(), mftc_train_purity_df["processed"].tolist())

subversion_non_moral_sim = simcse_model.similarity(mftc_train_subversion_df["processed"].tolist(), mftc_train_non_moral_df["processed"].tolist())



print("subversion - care: ", subversion_care_sim.mean())
print("subversion - fairness: ", subversion_fairness_sim.mean())
print("subversion - loyalty: ", subversion_loyalty_sim.mean())
print("subversion - authority: ", subversion_authority_sim.mean())
print("subversion - purity: ", subversion_purity_sim.mean())

print("subversion - harm: ", subversion_harm_sim.mean())
print("subversion - cheating: ", subversion_cheating_sim.mean())
print("subversion - betrayal: ", subversion_betrayal_sim.mean())
print("subversion - subversion: ", subversion_subversion_sim.mean())
print("subversion - degradation: ", subversion_degradation_sim.mean())


print("subversion - non-moral: ", subversion_non_moral_sim.mean())

subversion_vec_train = [subversion_care_sim.mean(), subversion_fairness_sim.mean(), subversion_loyalty_sim.mean(), subversion_authority_sim.mean(), subversion_purity_sim.mean(),
                 subversion_harm_sim.mean(), subversion_cheating_sim.mean(), subversion_betrayal_sim.mean(), subversion_subversion_sim.mean(), subversion_degradation_sim.mean(), subversion_non_moral_sim.mean()]

np.save("morality_table_mftc/subversion_mftc_all.npy", subversion_vec_train)


# In[ ]:


# degradation

degradation_harm_sim = simcse_model.similarity(mftc_train_degradation_df["processed"].tolist(), mftc_train_harm_df["processed"].tolist())
degradation_cheating_sim = simcse_model.similarity(mftc_train_degradation_df["processed"].tolist(), mftc_train_cheating_df["processed"].tolist())
degradation_betrayal_sim = simcse_model.similarity(mftc_train_degradation_df["processed"].tolist(), mftc_train_betrayal_df["processed"].tolist())
degradation_subversion_sim = simcse_model.similarity(mftc_train_degradation_df["processed"].tolist(), mftc_train_subversion_df["processed"].tolist())
degradation_degradation_sim = simcse_model.similarity(mftc_train_degradation_df["processed"].tolist(), mftc_train_degradation_df["processed"].tolist())

degradation_care_sim = simcse_model.similarity(mftc_train_degradation_df["processed"].tolist(), mftc_train_care_df["processed"].tolist())
degradation_fairness_sim = simcse_model.similarity(mftc_train_degradation_df["processed"].tolist(), mftc_train_fairness_df["processed"].tolist())
degradation_loyalty_sim = simcse_model.similarity(mftc_train_degradation_df["processed"].tolist(), mftc_train_loyalty_df["processed"].tolist())
degradation_authority_sim = simcse_model.similarity(mftc_train_degradation_df["processed"].tolist(), mftc_train_authority_df["processed"].tolist())
degradation_purity_sim = simcse_model.similarity(mftc_train_degradation_df["processed"].tolist(), mftc_train_purity_df["processed"].tolist())

degradation_non_moral_sim = simcse_model.similarity(mftc_train_degradation_df["processed"].tolist(), mftc_train_non_moral_df["processed"].tolist())




print("degradation - care: ", degradation_care_sim.mean())
print("degradation - fairness: ", degradation_fairness_sim.mean())
print("degradation - loyalty: ", degradation_loyalty_sim.mean())
print("degradation - authority: ", degradation_authority_sim.mean())
print("degradation - purity: ", degradation_purity_sim.mean())

print("degradation - harm: ", degradation_harm_sim.mean())
print("degradation - cheating: ", degradation_cheating_sim.mean())
print("degradation - betrayal: ", degradation_betrayal_sim.mean())
print("degradation - subversion: ", degradation_subversion_sim.mean())
print("degradation - degradation: ", degradation_degradation_sim.mean())


print("degradation - non-moral: ", degradation_non_moral_sim.mean())

degradation_vec_train = [degradation_care_sim.mean(), degradation_fairness_sim.mean(), degradation_loyalty_sim.mean(), degradation_authority_sim.mean(), degradation_purity_sim.mean(),
                  degradation_harm_sim.mean(), degradation_cheating_sim.mean(), degradation_betrayal_sim.mean(), degradation_subversion_sim.mean(), degradation_degradation_sim.mean(), degradation_non_moral_sim.mean()]

np.save("morality_table_mftc/degradation_mftc_all.npy", degradation_vec_train)


# In[ ]:


# non-morals

non_moral_harm_sim = simcse_model.similarity(mftc_train_non_moral_df["processed"].tolist(), mftc_train_harm_df["processed"].tolist())
non_moral_cheating_sim = simcse_model.similarity(mftc_train_non_moral_df["processed"].tolist(), mftc_train_cheating_df["processed"].tolist())
non_moral_betrayal_sim = simcse_model.similarity(mftc_train_non_moral_df["processed"].tolist(), mftc_train_betrayal_df["processed"].tolist())
non_moral_subversion_sim = simcse_model.similarity(mftc_train_non_moral_df["processed"].tolist(), mftc_train_subversion_df["processed"].tolist())
non_moral_degradation_sim = simcse_model.similarity(mftc_train_non_moral_df["processed"].tolist(), mftc_train_degradation_df["processed"].tolist())

non_moral_care_sim = simcse_model.similarity(mftc_train_non_moral_df["processed"].tolist(), mftc_train_care_df["processed"].tolist())
non_moral_fairness_sim = simcse_model.similarity(mftc_train_non_moral_df["processed"].tolist(), mftc_train_fairness_df["processed"].tolist())
non_moral_loyalty_sim = simcse_model.similarity(mftc_train_non_moral_df["processed"].tolist(), mftc_train_loyalty_df["processed"].tolist())
non_moral_authority_sim = simcse_model.similarity(mftc_train_non_moral_df["processed"].tolist(), mftc_train_authority_df["processed"].tolist())
non_moral_purity_sim = simcse_model.similarity(mftc_train_non_moral_df["processed"].tolist(), mftc_train_purity_df["processed"].tolist())

non_moral_non_moral_sim = simcse_model.similarity(mftc_train_non_moral_df["processed"].tolist(), mftc_train_non_moral_df["processed"].tolist())




print("non_moral - care: ", non_moral_care_sim.mean())
print("non_moral - fairness: ", non_moral_fairness_sim.mean())
print("non_moral - loyalty: ", non_moral_loyalty_sim.mean())
print("non_moral - authority: ", non_moral_authority_sim.mean())
print("non_moral - purity: ", non_moral_purity_sim.mean())

print("non_moral - harm: ", non_moral_harm_sim.mean())
print("non_moral - cheating: ", non_moral_cheating_sim.mean())
print("non_moral - betrayal: ", non_moral_betrayal_sim.mean())
print("non_moral - subversion: ", non_moral_subversion_sim.mean())
print("non_moral - degradation: ", non_moral_degradation_sim.mean())


print("non_moral - non-moral: ", non_moral_non_moral_sim.mean())

non_moral_vec_train = [non_moral_care_sim.mean(), non_moral_fairness_sim.mean(), non_moral_loyalty_sim.mean(), non_moral_authority_sim.mean(), non_moral_purity_sim.mean(),
                  non_moral_harm_sim.mean(), non_moral_cheating_sim.mean(), non_moral_betrayal_sim.mean(), non_moral_subversion_sim.mean(), non_moral_degradation_sim.mean(), non_moral_non_moral_sim.mean()]

np.save("morality_table_mftc/non_moral_mftc_all.npy", non_moral_vec_train)

