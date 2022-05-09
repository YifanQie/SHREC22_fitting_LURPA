# SHREC22_fitting_LURPA
SHREC 2022: Fitting and recognition of simple geometric primitives on point clouds

This is a solution to SHREC 2022: fitting track

Provided by Yifan QIE, LURPA, ENS Paris Saclay, Université Paris-Saclay

The files include:

1. load the point cloud from the SHREC22 track using main_SHREC22_dataprepare.m. The point clouds will be saved as images. (one-by-one for the purpose of checking)
2. load the all the 46000 point clouds from the SHREC22 track using DL_SHREC22_dataprepare.m and save them as images.
3. dl_SHREC_PCA_v2.m is used for traing the ALexNet using the generated 46000 images. (95%-5% splits)
4. my_net_trained_opti.mat is the parameters obtained after training
5. SHREC22Fitting.m loads the parameters for fine-tuned AlexNet and use it for surface type identification; Than a fitting process is conducted according to the identification results.
