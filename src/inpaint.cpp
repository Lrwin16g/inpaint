#include <iostream>
#include <vector>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "gFoE.h"

// インペイントの実行プログラム
int main( int argc, char **argv )
{
    if ( argc != 6 )
    {
        std::cerr << "Usage: " << argv[ 0 ] << " <input_img> <seed_img> <output_img> <param> <epsilon>" << std::endl;
        return -1;
    }
    
    // 入力画像とラベル画像の読込み
    cv::Mat srcImage = cv::imread( argv[ 1 ], 0 );
    cv::Mat seedImage = cv::imread( argv[ 2 ], 0 );
    
    // 入力画像を実数値に変換
    cv::Mat srcImage_64;
    srcImage.convertTo( srcImage_64, CV_64FC1 );
    
    // vectorに格納。値を0.0〜1.0へ変換
    std::vector<double> src;
    srcImage_64 = srcImage_64.reshape( 0, 1 ) / 255.0;
    srcImage_64.copyTo( src );
    
    // vectorに格納。黒色画素が補修箇所
    std::vector<unsigned char> mask;
    seedImage = ~seedImage.reshape( 0, 1 ) / 0xff;
    seedImage.copyTo( mask );

    // パラメータの読込み
    GaussFoE model;
    model.loadParameter( argv[ 4 ] );
    
    // インペイント実行
    std::vector<double> dst = model.inpaint( src, mask, 1.4, atof( argv[ 5 ] ), 1000000, srcImage.cols, srcImage.rows );
    
    // 値を0〜255へ変換
    cv::Mat dstImage_64( dst, true );
    dstImage_64 = dstImage_64.reshape( 1, srcImage.cols ) * 255.0;
    
    // 結果を保存
    cv::Mat dstImage;
    dstImage_64.convertTo( dstImage, CV_8UC1 );
    cv::imwrite( argv[ 3 ], dstImage );
        
    return 0;    
}
