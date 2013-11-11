#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "gFoE.h"

// GaussFoEの学習プログラム
int main( int argc, char **argv )
{
    if( argc != 5 )
    {
        std::cerr << "Usage: " << argv[ 0 ] << " <input> <output> <eta> <epsilon>" << std::endl;
        return -1;
    }
    
    // 学習用画像パスのリストを読込み
    std::ifstream ifs( argv[ 1 ] );
    if( ifs.is_open() == false )
    {
        std::cerr << "File open error: " << argv[ 1 ] << std::endl;
        return -1;
    }
    
    std::string str;
    std::vector< std::string > filelist;
    while( getline( ifs, str ) )
    {
        filelist.push_back( str );
    }
    ifs.close();
    
    // 学習用画像をvectorに格納
    int width = 16;
    int height = 16;
    std::vector< std::vector<double> > supervisor;
    for( size_t i = 0; i < filelist.size(); ++i )
    {
        cv::Mat_<double> image = cv::imread( filelist[ i ].c_str(), 0 );
        image = image.reshape( 0, 1 ) / 255.0;
        
        std::vector<double> value;
        image.copyTo( value );
        supervisor.push_back( value );
    }

    // 学習
    GaussFoE model( 8, 3, 3 );
    model.train( supervisor, atof( argv[ 3 ] ), atof( argv[ 4 ] ), 1000000, width, height );
    model.saveParameter( argv[ 2 ] );
    
    return 0;
}
