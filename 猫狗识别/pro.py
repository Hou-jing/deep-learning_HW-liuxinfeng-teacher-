import os, shutil

original_dataset_dir = './train' #原数据位置

base_dir = './dogs_and_cats_small_2' #选取的数据的新位置
os.mkdir(base_dir) #创建新位置文件夹

#在新文件夹中依次新增‘train’，‘validation’，‘test’文件夹
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

#在‘train’文件夹中新增‘dogs’，‘cats’文件夹
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

#在‘validation’文件夹中新增‘dogs’，‘cats’文件夹
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

#在‘test’文件夹新增‘dogs’，‘cats’文件夹
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

#将原数据中前1000张猫的图片复制到train_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname) #原位置
    dst = os.path.join(train_cats_dir, fname) #目标位置
    shutil.copyfile(src, dst) #复制文件

#将原数据中1000-1500张猫的图片复制到validation_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(2000, 2500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

#将原数据中1500-2000张猫的图片复制到test_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(2500, 3000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

#将原数据中前1000张狗的图片复制到train_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

#将原数据中1000-1500张狗的图片复制到validation_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(2000,2500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

#将原数据中1500-2000张狗的图片复制到test_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(2500,3000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)
