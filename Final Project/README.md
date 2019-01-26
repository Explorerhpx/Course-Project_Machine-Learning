# Final Project:  Identification and Synthesizing of Raphael’s paintings from the forgeries
This Project is the Final Project of DATA130003

This project is implemented by Python.

## Remark
* This Project is a group work implemented by Pingxuan Huang, Xinzhe Luo, Hanwen Wu. Codes are available, if you want to utilize them, however, please indicate the source;
* Dataset is  provided by Prof. Yang WANG from [HKUST]( https://drive.google.com/folderview?id=0B-yDtwSjhaSCZ2FqN3AxQ3NJNTA&usp=sharing);
* The labels file of paintings is [here]( https://docs.google.com/document/d/1tMaaSIrYwNFZZ2cEJdx1DfFscIfERd5Dp2U7K1ekjTI/edit);
* If you have any question, please don't hesitate to contact *1336076538@qq.com* for help.
* Some related papers are provided as reference (you can obtain them at the [*References*](#reference) part);

## Introduction
### Requirement
This project could be divided into 2 separated tasks:
    1.   Classification task: students are required to distinguish the authentic Raphael paintings from these fake paintings;
    2.   Neural Style Transfer task: students  are required to transfer the style of Raphael’s paintings to other photos.

Please check *Requirement* for further information about this project.

### Our work
* For identification problem, we first use Geometric tight frame and Gabor wavelet to extract features. With selected feature based on area under ROC curve(AUC), we apply several techniques such as SVM, Neural-Network and Decision-Tree for classification. Leave-one-out cross validation is used to ensure accuracy.
* As for the synthesizing problem, with the help of VGG network and the well-defined content & style loss functions, style transfer is reduced into an optimization problem.

Please check *Report* for the detail information about the models and our experiment.

## <span id="reference"> References </span>
\[1\] G. Berger and R. Memisevic. Incorporating long-range consistency in cnn-based texture generation. CoRR, abs/1606.01286, 2016.

\[2\] L. A. Gatys, A. S. Ecker, and M. Bethge. A neural algorithm of artistic style. CoRR, abs/1508.06576, 2015.

\[3\] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. CoRR, abs/1512.03385, 2015.

\[4\] P. Isola, J. Zhu, T. Zhou, and A. A. Efros. Image-to-image translation with conditional adversarial networks. CoRR, abs/1611.07004, 2016.

\[5\] C. R. Johnson, E. Hendriks, I. Berezhnoy, E. Brevdo, S. Hughes, I. Daubechies, J. Li, E. Postma, and J. Z. Wang. Image processing for artist identification – computerized analysis of Vincent van gogh’s painting brushstrokes. IEEE Signal Processing Magazine, July 2008.

\[6\] J. Johnson, A. Alahi, and F. Li. Perceptual losses for real-time style transfer and super-resolution. CoRR, abs/1603.08155, 2016.

\[7\] J. Li, L. Yao, E. Hendriks, and J. Z. Wang. Rhythmic brushstrokes distinguish van gogh from his contemporaries: Findings via automated brushstroke extraction. IEEE Trans. Pattern Anal. Mach. Intell., 34(6):1159–1176, 2012.

\[8\] J. Liao, Y. Yao, L. Yuan, G. Hua, and S. B. Kang. Visual attribute transfer through deep image analogy. CoRR, abs/1705.01088, 2017.

\[9\] H. Liu, R. H. Chan, and Y. Yao. Geometric tight frame based stylometry for art authentication of van gogh paintings. CoRR, abs/1407.0439, 2014.

\[10\] A. Mahendran and A. Vedaldi. Understanding deep image representations by inverting them. CoRR, abs/1412.0035, 2014.

\[11\] H. Qi, A. Taeb, and S. M. Hughes. Visual stylometry using background selection and wavelet- hmt-based fisher information distances for attribution and dating of impressionist paintings.  Signal Processing, 93(3):541–553, 2013.

\[12\] A. Radford, L. Metz, and S. Chintala. Unsupervised representation learning with deep convolu- tional generative adversarial networks. CoRR, abs/1511.06434, 2015.














