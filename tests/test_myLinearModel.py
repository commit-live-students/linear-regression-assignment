from unittest import TestCase


class TestMyLinearModel(TestCase):
    def test_load_data(self):
        from build import load_data
        X, y = load_data()
        self.assertItemsEqual([506,13], X.shape, msg='feature matrix has incorrect dimensions')
        self.assertEqual(y.shape[0], 506, msg='target has incorrect dimensions')

    def test_myLinearModel(self):
        from build import myLinearModel, load_data
        X, y = load_data()
        mse_lr, _ = myLinearModel(X, y)
        self.assertAlmostEqual(mse_lr, 21.897779217687496, 2)

    def test_myRidge(self):
        from build import myRidge, load_data
        X, y = load_data()
        mse_ridge, _ = myRidge(X, y, 0.1)
        self.assertAlmostEqual(mse_ridge, 21.901368882210598, 2)

    def test_myLasso(self):
        from build import myLasso, load_data
        X, y = load_data()
        mse_lasso, _ = myLasso(X, y, 0.1)
        self.assertAlmostEqual(mse_lasso, 23.053740394929395, 2)

    def test_polynomial_lr_pipeline(self):
        from build import polynomial_lr_pipeline, load_data
        X, y = load_data()
        mse_plr, _ = polynomial_lr_pipeline(X, y, 3)
        self.assertAlmostEqual(mse_plr, 0.17022286697767638, 2)

    def test_polynomial_redge_pipeline(self):
        from build import polynomial_redge_pipeline, load_data
        X, y = load_data()
        mse_pr, _ = polynomial_redge_pipeline(X, y, 2, 0.01)
        self.assertAlmostEqual(mse_pr, 6.0111147238206195, 2)

    def test_cross_val(self):
        from build import cross_val, load_data
        X, y = load_data()
        mse_cv = cross_val(X, y, 2, 0.1, 100)
        self.assertAlmostEqual(mse_cv, -16.731323379923818, 2)
