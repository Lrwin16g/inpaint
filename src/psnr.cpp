#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

double psnr(const cv::Mat &src_img, const cv::Mat &cmp_img)
{
    // グレースケールに変換
    cv::Mat src_gray, cmp_gray;
    cv::cvtColor(src_img, src_gray, CV_BGR2GRAY);
    cv::cvtColor(cmp_img, cmp_gray, CV_BGR2GRAY);
    
    // 符号あり16bitに変換
    cv::Mat src_16s, cmp_16s;
    src_gray.convertTo(src_16s, CV_16S);
    cmp_gray.convertTo(cmp_16s, CV_16S);
    
    // MSE, PSNRを計算
    cv::Mat sub_16s = src_16s - cmp_16s;
    double sum_ = static_cast<double>(sum(sub_16s.mul(sub_16s))[0]);
    double mse = sum_ / (src_16s.rows * src_16s.cols);
    double psnr = 10.0 * log10(255.0 * 255.0 / mse);
    
    return psnr;
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
	std::cerr << "Usage: " << argv[0] << " <image_1> <image_2>" << std::endl;
	return -1;
    }
    
    cv::Mat src_img = cv::imread(argv[1]);
    cv::Mat cmp_img = cv::imread(argv[2]);
    
    std::cout << "PSNR=" << psnr(src_img, cmp_img) << "[dB]" << std::endl;
    
    return 0;
}
