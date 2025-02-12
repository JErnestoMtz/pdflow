#%%
import pdflow

#%%
test_pdf = './test1.pdf'
qr_codes = pdflow.extract_qrs(test_pdf)
qr_codes[0][0]

# %%
segmentation_model = pdflow.get_segmentation_model()
# %%
