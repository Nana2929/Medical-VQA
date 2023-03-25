import _pickle as cPickle
images_path = '/home/nanaeilish/projects/mis/PubMedCLIP/QCR_PubMedCLIP/data/data_rad/images250x250.pkl'
clip_images_data = cPickle.load(open(images_path, 'rb'))
print(len(clip_images_data))
# find . -name "*.jpg" | wc -l