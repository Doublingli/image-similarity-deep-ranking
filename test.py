import deepranking_get_distance as dgd

model_path = "./deepranking.h5"

model = dgd.deep_rank_model()
model.load_weights(model_path)

for i in range(1,4):
    image1 = dgd.load_image("./testset/t" + str(i) + "-1.jpg")
    image2 = dgd.load_image("./testset/t" + str(i) + "-2.jpg")
    distance = dgd.compare(model, image1, image2)
    print("Distance%i:%s" % (i,distance))
