## DNN or Dermatologist Supporting code:
 * This repository contains all necessary code to replicate the experiments produced in the paper which can be seen below on Arxiv.
 * A suggested approach is to start with the "train_and_test_set_creation.ipynb" to recreate the training and testing sets used before training a suite of models using the "DataSplit_HpSearch.py" in which a hyperparameter search is also undertaken. KernelSHAP and GradCAM explanations can then be produced using files contained in the "Shap_GradCAM_Notebooks" directory. Our data analysis code for the sanity checks described in the paper is then contained in "Model_Sensitivity_Experiments" and "Randomised_layer_experiments". 


*Arxiv*: http://arxiv.org/abs/1908.06612

Sample bibtex file goes here 

## References
* [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567) (CVPR 2016)
* Tschandl, P., Rosendahl, C., Kittler, H.: The ham10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific data 5, 180161 (2018)
* Lundberg, S.M., Lee, S.I.: A unified approach to interpreting model predictions. In: Advances in Neural Information Processing Systems. pp. 4765–4774 (2017), https://github.com/slundberg/shap
* Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D.: Gradcam: Visual explanations from deep networks via gradient-based localization. In: Proceedings of the IEEE International Conference on Computer Vision. pp. 618–626 (2017)
* Kotikalapudi, Raghavendra: keras-vis, https://github.com/raghakot/keras-vis
