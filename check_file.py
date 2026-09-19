import nibabel as nib
img = nib.load(r"C:\Ischemic Stroke Lesion Segmentation\isles2022\ISLES\dataset\derivatives\sub-strokecase0001\ses-0001\sub-strokecase0001_ses-0001_msk.nii.gz")
print("Размер:", img.shape)
print("Уникальные значения:", set(img.get_fdata().ravel()))