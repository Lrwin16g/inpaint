#ifndef INPAINT_GFOE_H_
#define INPAINT_GFOE_H_

#include <vector>

// Gaussian Field of Expertsクラス
class GaussFoE
{    
public:
    GaussFoE();
    GaussFoE(int expertsNum, int cliqueWidth, int cliqueHeight);
    ~GaussFoE();
    
    void train(const std::vector<std::vector<double> > &sample,
               double eta, double epsilon, int maxLoop,
               int width, int height);
    
    std::vector<double> inpaint(const std::vector<double> &src,
                                const std::vector<unsigned char> &mask,
                                double omega, double epsilon, int maxLoop,
                                int width, int height);
    
    void loadParameter(const char *filename);
    void saveParameter(const char *filename);
    
private:
    void initialize();
    void cleanUp();
    void defineClique(std::vector<std::vector<int> > &pixelIndexArrayInClique,
                      std::vector<std::vector<int> > &cliqueIndexArrayInPixel,
                      int width, int height);
    int  getPixelIndexInClique(int pixelIndex, const std::vector<int> &pixelIndexArray);
    inline int cliqueSize() const {return cliqueWidth_ * cliqueHeight_;};
    
    double	**alpha_;
    double   	*beta_;
    double      lambda_;
    int         expertsNum_;
    int         cliqueWidth_;
    int         cliqueHeight_;
    
    // DISALLOW_COPY_AND_ASSIGN
    GaussFoE(const GaussFoE&);
    void operator=(const GaussFoE&);
};

void inverseMatrix(double const * const * src, double **dst, int dim);

#endif	// INPAINT_GFOE_H_
