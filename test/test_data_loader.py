# 测试数据读取

from data.data_loader import TBMDataset,TBMDataset_Pred

class TestTBMDataset():
    def test_init_dataset(self):
        Test_Dataset = TBMDataset()
        assert Test_Dataset is not None
    
    def test_get_data(self):
        Test_Dataset = TBMDataset()
        print('\ndata_shape:',Test_Dataset.data_x.shape)
        print('\nlabel_shape:',Test_Dataset.data_y.shape)

    def test_get_train_data(self):
        Test_Dataset = TBMDataset()
        t,l  = Test_Dataset[0]
        print('\ndata_size:', t.shape)
        print('\nlabel_size:', l.shape)

    def test_get_vali_data(self):
        Test_Dataset = TBMDataset(flag='val')
        t,l  = Test_Dataset[0]
        print('\ndata_size:' ,t.shape,' ',t[0])
        print('\nlabel_size:' ,l.shape)

    def test_get_test_data(self):
        T_DS = TBMDataset(flag='test')
        t,l = T_DS[0]
        t1,l1 = T_DS[1]
        print('\ndata_size:' ,t.shape,' ',t[0])
        print('\ndata_size:' ,t1.shape,' ',t1[0])
        print('\nlabel_size:' ,l.shape)

    def test_get_predict_data(self):
        T_DS_p = TBMDataset_Pred()
        t,l = T_DS_p[0]
        print('\ndata_size:' ,t.shape,' ',t[0])
        print('\nlabel_size:' ,l.shape)