#ifndef _GFOE_H
#define _GFOE_H

#include <vector>

// Gaussian Field of Expertsクラス
class GaussFoE
{    
public:
    GaussFoE();
    GaussFoE( int expertsNum, int cliqueWidth, int cliqueHeight );
    ~GaussFoE();
    
    void train( const std::vector< std::vector<double> > &sample,
                double eta, double epsilon, int maxLoop,
                int width, int height );
    
    std::vector<double> inpaint( const std::vector<double> &src,
                                 const std::vector<unsigned char> &mask,
                                 double omega, double epsilon, int maxLoop,
                                 int width, int height );
    
    inline void setAlpha( int i, int j, double value )    { alpha_[ i ][ j ] = value; };
    inline void setBeta( int i, double value )            { beta_[ i ] = value; };
    inline void setLambda( double value )                 { lambda_ = value; };
    
    inline double   alpha( int i, int j ) const { return alpha_[ i ][ j ]; };
    inline double   beta( int i )         const { return beta_[ i ]; };
    inline double   lambda()              const { return lambda_; };
    inline int      expertsNum()          const { return expertsNum_; };
    inline int      cliqueWidth()         const { return cliqueWidth_; };
    inline int      cliqueHeight()        const { return cliqueHeight_; };
    inline int      cliqueSize()          const { return cliqueWidth_ * cliqueHeight_; };

    void loadParameter( const char *filename );
    void saveParameter( const char *filename );

private:
    void initialize();
    void cleanUp();
    void defineClique( std::vector< std::vector<int> > &pixelIndexArrayInClique,
                       std::vector< std::vector<int> > &cliqueIndexArrayInPixel,
                       int width, int height );
    int  getPixelIndexInClique( int pixelIndex, const std::vector<int> &pixelIndexArray );
    void inverseMatrix( double const * const * src, double **dst, int dim );
    
    double**    alpha_;
    double*     beta_;
    double      lambda_;
    int         expertsNum_;
    int         cliqueWidth_;
    int         cliqueHeight_;
    
    // DISALLOW_COPY_AND_ASSIGN
    GaussFoE( const GaussFoE& );
    void operator=( const GaussFoE& );
};

#endif
