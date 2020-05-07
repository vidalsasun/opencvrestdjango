class PyTorchtranslateParams:
    trained_model = 'weights/craft_mlt_25k.pth'
    text_threshold = 0.7
    low_text = 0.4
    link_threshold = 0.4
    cuda = False
    canvas_size = 1280
    mag_ratio = 1.5
    poly = False
    show_time = False
    test_folder = '/data/'
    refine = False
    refiner_model = 'weights/craft_refiner_CTW1500.pth'
