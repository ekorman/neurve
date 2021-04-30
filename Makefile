data/car_ims.tgz :
	curl http://imagenet.stanford.edu/internal/car196/car_ims.tgz --output data/car_ims.tgz

data/car_ims : data/car_ims.tgz
	tar -xzf data/car_ims.tgz -C data/

data/cars_annos.mat :
	curl http://imagenet.stanford.edu/internal/car196/cars_annos.mat --output data/cars_annos.mat

data/cars : data/car_ims data/cars_annos.mat
	python data/make_car_folders.py
	rmdir data/car_ims
