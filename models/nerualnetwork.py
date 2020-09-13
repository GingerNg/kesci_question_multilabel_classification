from keras.models import Sequential
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D, Dropout
from keras.models import load_model

class MultiLabelModel(object):
    def __init__(self):
        self.model = None
        self.model_path = None

    def load_model(self):
        if self.model is None:
            self.model = load_model(self.model_path)

    def INITModel(self,feature_dim,label_dim):

        model = Sequential()
        print("create model. feature_dim = %s, label_dim = %s" % (feature_dim,label_dim))
        model.add(Dense(300, activation='relu', input_dim=feature_dim))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(label_dim, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # plot_model(model, to_file='multi_label_model.png')
        self.model = model
        # return model

    def train(self,trainX,trainY):
        # model = self.INITModel(feature_dim=80, label_dim=67)
        history = self.model.fit(trainX, trainY, epochs=50, verbose=0)
        # self.model = model
        if self.model_path:
            model.save(self.model_path)
        # return model

    def test(self,testX,testY):
        """[Keras中model.evaluate（）返回的是 损失值和你选定的指标值（例如，精度accuracy）。]

        Args:
            testX ([type]): [description]
            testY ([type]): [description]

        Returns:
            [type]: [description]
        """
        score = self.model.evaluate(testX, testY, verbose=0)
        return score

    def output(self,line):
        # print(self.model.predict(line))
        return self.model.predict_classes(line)






