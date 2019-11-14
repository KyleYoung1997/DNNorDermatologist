## DNN or Dermatologist Supporting code:
 * This repository contains all necessary code to replicate the experiments in Deep Neural Networks or Dermatologists?: http://arxiv.org/abs/1908.06612.
 * We suggest to start with train_and_test_set_creation.ipynb. 
 * Then train a suite of models using the DataSplit_HpSearch.py file. 
 * Then produce Kernel SHAP and GradCAM explanations: "Shap_GradCAM_Notebooks". 
 * Sanity check code is contained in "Model_Sensitivity_Experiments"	and "Randomised_layer_experiments"
	


Arxiv: http://arxiv.org/abs/1908.06612

Sample Bibtex file: 

@InProceedings{10.1007/978-3-030-33850-3_6,

author="Young, Kyle

and Booth, Gareth

and Simpson, Becks

and Dutton, Reuben

and Shrapnel, Sally",

editor="Suzuki, Kenji

and Reyes, Mauricio

and Syeda-Mahmood, Tanveer

and Glocker, Ben

and Wiest, Roland
and Gur, Yaniv

and Greenspan, Hayit

and Madabhushi, Anant",

title="Deep Neural Network or Dermatologist?",

booktitle="Interpretability of Machine Intelligence in Medical Image Computing and Multimodal Learning for Clinical Decision Support",

year="2019",

publisher="Springer International Publishing",

address="Cham",

pages="48--55",

isbn="978-3-030-33850-3"
}
   


## References
* [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567) (CVPR 2016)
* Tschandl, P., Rosendahl, C., Kittler, H.: The ham10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific data 5, 180161 (2018)
* Lundberg, S.M., Lee, S.I.: A unified approach to interpreting model predictions. In: Advances in Neural Information Processing Systems. pp. 4765–4774 (2017), https://github.com/slundberg/shap
* Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D.: Gradcam: Visual explanations from deep networks via gradient-based localization. In: Proceedings of the IEEE International Conference on Computer Vision. pp. 618–626 (2017)
* Kotikalapudi, Raghavendra: keras-vis, https://github.com/raghakot/keras-vis
