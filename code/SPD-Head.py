class SpaceToDepth(tf.keras.layers.Layer):
    def __init__(self, dimension=1):
        super(SpaceToDepth, self).__init__()
        self.d = dimension

    def call(self, x):
        height, width = x.shape[1], x.shape[2]
        x = tf.reshape(x, [-1, height // 2, 2, width // 2, 2, x.shape[3]])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [-1, height, width, 1 * x.shape[5]])
        return x

def body(input_shape, num_classes,weight_decay=5e-4):

    fpn_outs = [P3, P4, P5]
    outs = []
    for i, out in enumerate(fpn_outs):
        # 利用1x1卷积进行通道整合
        stem = DarknetConv2D_BN_Leaky(int(256 * 0.375), (1, 1), strides=(1, 1), weight_decay=weight_decay,
                                     name='head.stems.' + str(i))(out)
        # # # 利用3x3卷积进行特征提取
        cls_conv = DarknetConv2D_BN_Leaky(int(256 * 0.375), (3, 3), strides=(1, 1), weight_decay=weight_decay,
                                         name='head.cls_convs.' + str(i) + '.0')(stem)
        spd=SpaceToDepth()(stem)
        cls_conv = DarknetConv2D_BN_Leaky(int(256 * 0.375), (3, 3), strides=(1, 1), weight_decay=weight_decay,
                                         name='head.cls_convs.' + str(i) + '.1')(cls_conv)
        cls_conv=Concatenate(axis=-1)([spd,cls_conv])
        # ---------------------------------------------------#
        #   判断特征点所属的种类
        #   80, 80, num_classes
        #   40, 40, num_classes
        #   20, 20, num_classes
        # ---------------------------------------------------#
        cls_pred = DarknetConv2D(num_classes, (1, 1), strides=(1, 1), weight_decay=weight_decay,
                                 name='head.cls_preds.' + str(i))(cls_conv)

        # 利用3x3卷积进行特征提取
        reg_conv = DarknetConv2D_BN_Leaky(int(256 * 0.375), (3, 3), strides=(1, 1), weight_decay=weight_decay,
                                         name='head.reg_convs.' + str(i) + '.0')(stem)
        spd2=SpaceToDepth()(stem)
        reg_conv = DarknetConv2D_BN_Leaky(int(256 * 0.375), (3, 3), strides=(1, 1), weight_decay=weight_decay,
                                         name='head.reg_convs.' + str(i) + '.1')(reg_conv)
        reg_conv=Concatenate(axis=-1)([spd2,reg_conv])
        # ---------------------------------------------------#
        #   特征点的回归系数
        #   reg_pred 80, 80, 4
        #   reg_pred 40, 40, 4
        #   reg_pred 20, 20, 4
        # ---------------------------------------------------#
        reg_pred = DarknetConv2D(4, (1, 1), strides=(1, 1), weight_decay=weight_decay, name='head.reg_preds.' + str(i))(
            reg_conv)
        # ---------------------------------------------------#
        #   判断特征点是否有对应的物体
        #   obj_pred 80, 80, 1
        #   obj_pred 40, 40, 1
        #   obj_pred 20, 20, 1
        # ---------------------------------------------------#
        obj_pred = DarknetConv2D(1, (1, 1), strides=(1, 1), weight_decay=weight_decay, name='head.obj_preds.' + str(i))(
            reg_conv)
        output = Concatenate(axis=-1)([reg_pred, obj_pred, cls_pred])
        outs.append(output)
    return Model(inputs, outs)