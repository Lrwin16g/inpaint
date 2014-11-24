#include "gFoE.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>

GaussFoE::GaussFoE()
    : expertsNum_(0), cliqueWidth_(0), cliqueHeight_(0),
      alpha_(NULL), beta_(NULL), lambda_(0.0)
{
    srand(time(NULL));
}

GaussFoE::GaussFoE(int expertsNum, int cliqueWidth, int cliqueHeight)
    : expertsNum_(expertsNum), cliqueWidth_(cliqueWidth), cliqueHeight_(cliqueHeight),
      alpha_(NULL), beta_(NULL), lambda_(0.0)
{
    srand(time(NULL));
    initialize();
}

GaussFoE::~GaussFoE()
{
    cleanUp();
}

// 学習
void GaussFoE::train(const std::vector<std::vector<double> > &sample,	// 学習サンプル
                     double eta,	// 学習率
                     double epsilon,	// 終了条件
                     int    maxLoop,	// 最大反復回数
                     int    width,	// クリークの幅
                     int    height)	// クリークの高さ
{
    // 学習パラメータの初期化
    lambda_ = static_cast<double>(rand()) / RAND_MAX;
    for (int i = 0; i < cliqueSize(); ++i)
    {
        beta_[i] = static_cast<double>(rand()) / RAND_MAX / 10.0 * pow(-1.0, rand() % 2);
        for (int j = 0; j < expertsNum_; ++j)
	{
            alpha_[j][i] = static_cast<double>(rand()) / RAND_MAX * 10.0 * pow(-1.0, rand() % 2);
        }
    }
    
    std::vector<std::vector<double> > supervisor(sample.size());
    for (size_t i = 0; i < sample.size(); ++i)
    {
        std::copy(sample[i].begin(), sample[i].end(), back_inserter(supervisor[i]));
    }
    
    // クリークのサイズから画素とクリークを対応付ける
    std::vector<std::vector<int> > pixelIndexArrayInClique;
    std::vector<std::vector<int> > cliqueIndexArrayInPixel;
    defineClique(pixelIndexArrayInClique, cliqueIndexArrayInPixel, width, height);
    
    int pixelSize = width * height;
    double **omega = new double*[pixelSize];
    double **omega_inv = new double*[pixelSize];
    for (int i = 0; i < pixelSize; ++i) {
        omega[i] = new double[pixelSize];
        omega_inv[i] = new double[pixelSize];
    }
    
    double *h = new double[pixelSize];
    double *mean = new double[pixelSize];
    double *beta_delta = new double[cliqueSize()];
    double *beta_update = new double[cliqueSize()];
    double **alpha_delta = new double*[expertsNum_];
    double **alpha_update = new double*[expertsNum_];
    for (int i = 0; i < expertsNum_; ++i) {
        alpha_delta[i] = new double[cliqueSize()];
        alpha_update[i] = new double[cliqueSize()];
    }
    
    double eta0 = eta;
    
    // 学習開始
    for (int loop = 0; loop < maxLoop; ++loop)
    {
        for (int i = 0; i < pixelSize; ++i)
	{
            for (int j = 0; j < pixelSize; ++j)
	    {
            	// 行列Ωの計算
                omega[i][j] = 0.0;
                std::vector<int> commonCliqueIndexArray;
                for (size_t k = 0; k < cliqueIndexArrayInPixel[i].size(); ++k)
		{
                    int cliqueIndex_1 = cliqueIndexArrayInPixel[i][k];
                    for (size_t l = 0; l < cliqueIndexArrayInPixel[j].size(); ++l)
		    {
                        int cliqueIndex_2 = cliqueIndexArrayInPixel[j][l];
                        if (cliqueIndex_1 == cliqueIndex_2)
			{
                            commonCliqueIndexArray.push_back(cliqueIndex_1);
                            break;
                        }
                    }
                }
                
                for (int k = 0; k < expertsNum_; ++k)
		{
                    for (size_t l = 0; l < commonCliqueIndexArray.size(); ++l)
		    {
                        int cliqueIndex = commonCliqueIndexArray[l];
                        omega[i][j] += (alpha_[k][getPixelIndexInClique(i, pixelIndexArrayInClique[cliqueIndex])]
                                        * alpha_[k][getPixelIndexInClique(j, pixelIndexArrayInClique[cliqueIndex])]);
                    }
                }
                
                if (i == j)
		{
                    omega[i][j] += (2.0 * lambda_ * static_cast<double>(cliqueIndexArrayInPixel[i].size()));
                }
            }
        }
        
        // hの計算
        for (int i = 0; i < pixelSize; ++i)
	{
            h[i] = 0.0;
            for (size_t j = 0; j < cliqueIndexArrayInPixel[i].size(); ++j)
	    {
                int cliqueIndex = cliqueIndexArrayInPixel[i][j];
                h[i] += beta_[getPixelIndexInClique(i, pixelIndexArrayInClique[cliqueIndex])];
            }
        }
        
        // 行列Ωの逆行列の計算
        inverseMatrix(omega, omega_inv, pixelSize);
        
        for (int i = 0; i < pixelSize; ++i)
	{
            mean[i] = 0.0;
            for (int j = 0; j < pixelSize; ++j)
	    {
                mean[i] += (omega_inv[i][j] * h[j]);
            }
        }
        
        // 学習サンプルの順序をランダムに並び替え
        for (size_t i = 0; i < supervisor.size(); ++i)
	{
            int index = rand() % supervisor.size();
            std::swap(supervisor[i], supervisor[index]);
        }
        
        // λの更新
        double term_1 = 0.0;
        for (size_t i = 0; i < supervisor.size(); ++i)
	{
            for (size_t j = 0; j < pixelIndexArrayInClique.size(); ++j)
	    {
                for (int k = 0; k < cliqueSize(); ++k)
		{
                    int pixelIndex = pixelIndexArrayInClique[j][k];
                    term_1 += pow(supervisor[i][pixelIndex], 2.0);
                }
            }
        }
        
        double term_2 = 0.0;
        for (size_t i = 0; i < pixelIndexArrayInClique.size(); ++i)
	{
            for (int j = 0; j < cliqueSize(); ++j)
	    {
                int pixelIndex = pixelIndexArrayInClique[i][j];
                term_2 += pow(mean[pixelIndex], 2.0);
            }
        }
        
        double lambda_delta = -1.0 / static_cast<double>(supervisor.size()) * term_1 + term_2;
        
        // βの更新
        for (int i = 0; i < cliqueSize(); ++i)
	{
            double term_3 = 0.0;
            for (size_t j = 0; j < supervisor.size(); ++j)
	    {
                for (size_t k = 0; k < pixelIndexArrayInClique.size(); ++k)
		{
                    int pixelIndex = pixelIndexArrayInClique[k][i];
                    term_3 += supervisor[j][pixelIndex];
                }
            }
            
            double term_4 = 0.0;
            for (size_t j = 0; j < pixelIndexArrayInClique.size(); ++j)
	    {
                int pixelIndex = pixelIndexArrayInClique[j][i];
                term_4 += mean[pixelIndex];
            }
            
            beta_delta[i] = 1.0 / static_cast<double>(supervisor.size()) * term_3 - term_4;
        }
        
        // αの更新
        for (int i = 0; i < expertsNum_; ++i)
	{
            for (int j = 0; j < cliqueSize(); ++j)
	    {
                double term_5 = 0.0;
                for (size_t k = 0; k < supervisor.size(); ++k)
		{
                    for (size_t l = 0; l < pixelIndexArrayInClique.size(); ++l)
		    {
                        int pixelIndex_2 = pixelIndexArrayInClique[l][j];
                        for (int m = 0; m < cliqueSize(); ++m)
			{
                            int pixelIndex_1 = pixelIndexArrayInClique[l][m];
                            term_5 += (alpha_[i][m] * supervisor[k][pixelIndex_1] * supervisor[k][pixelIndex_2]);
                        }
                    }
                }
                
                double term_6 = 0.0;
                for (size_t k = 0; k < pixelIndexArrayInClique.size(); ++k)
		{
                    int pixelIndex_2 = pixelIndexArrayInClique[k][j];
                    for (int l = 0; l < cliqueSize(); ++l)
		    {
                        int pixelIndex_1 = pixelIndexArrayInClique[k][l];
                        term_6 += (alpha_[i][l] * (omega_inv[pixelIndex_1][pixelIndex_2] + mean[pixelIndex_1] * mean[pixelIndex_2]));
                    }
                }
                
                alpha_delta[i][j] = -1.0 / static_cast<double>(supervisor.size()) * term_5 + term_6;
            }
        }
        
        double lambda_update = lambda_ + (eta * lambda_delta);
        for (int i = 0; i < cliqueSize(); ++i)
	{
            beta_update[i] = beta_[i] + (eta * beta_delta[i]);
            for (int j = 0; j < expertsNum_; ++j)
	    {
                alpha_update[j][i] = alpha_[j][i] + (eta * alpha_delta[j][i]);
            }
        }
        
        // 更新度合いを計算
        double diff = fabs(lambda_ - lambda_update);
        for (int i = 0; i < cliqueSize(); ++i)
	{
            diff += fabs(beta_[i] - beta_update[i]);
            for (int j = 0; j < expertsNum_; ++j)
	    {
                diff += fabs(alpha_[j][i] - alpha_update[j][i]);
            }
        }
        
        lambda_ = lambda_update;
        for (int i = 0; i < cliqueSize(); ++i)
	{
            beta_[i] = beta_update[i];
            for (int j = 0; j < expertsNum_; ++j)
	    {
                alpha_[j][i] = alpha_update[j][i];
            }
        }
        
        // 終了条件を確認
        if (diff < epsilon)
	{
            break;
        }
        
        eta -= (eta0 / static_cast<double>(maxLoop));
        
        if (loop % 10 == 0)
	{
            std::cout << loop << ": " << diff << std::endl;
            saveParameter("tmp.prm");
        }
    }
    
    for (int i = 0; i < pixelSize; ++i) {
        delete[] omega[i];
        delete[] omega_inv[i];
    }
    delete[] omega; omega = NULL;
    delete[] omega_inv; omega_inv = NULL;
    delete[] h; h = NULL;
    delete[] mean; mean = NULL;
    delete[] beta_delta; beta_delta = NULL;
    delete[] beta_update; beta_update = NULL;
    
    for (int i = 0; i < expertsNum_; ++i) {
        delete[] alpha_delta[i];
        delete[] alpha_update[i];
    }
    delete[] alpha_delta; alpha_delta = NULL;
    delete[] alpha_update; alpha_update = NULL;
    
}

// インペイント
std::vector<double> GaussFoE::inpaint(const std::vector<double>        &src,	// 入力画像
                                      const std::vector<unsigned char> &mask,	// ラベル画像
                                      double   omega,	// パラメータ
                                      double   epsilon,	// 終了条件
                                      int      maxLoop,	// 最大反復回数
                                      int      width,	// クリークの幅
                                      int      height)	// クリークの高さ
{
    std::vector<double> data;
    std::copy(src.begin(), src.end(), back_inserter(data));
    
    std::vector<int> target;
    for (size_t i = 0; i < src.size(); ++i)
    {
        if (mask[i] == 1)
	{
            target.push_back(i);
        }
    }
    
    // クリークのサイズから画素とクリークを対応付ける
    std::vector<std::vector<int> > pixelIndexArrayInClique;
    std::vector<std::vector<int> > cliqueIndexArrayInPixel;
    defineClique(pixelIndexArrayInClique, cliqueIndexArrayInPixel, width, height);
    
    // 補修対象画素を初期化
    for (size_t i = 0; i < target.size(); ++i)
    {
        int targetIndex = target[i];
        data[targetIndex] = static_cast<double>(rand()) / RAND_MAX;
    }
    
    double omega0 = omega;
    
    // 補修開始
    for (int loop = 0; loop < maxLoop; ++loop)
    {
        double distortion = 0.0;
        for (size_t i = 0; i < target.size(); ++i)
	{
            int targetIndex = target[i];
            double term_1 = static_cast<double>(cliqueIndexArrayInPixel[targetIndex].size()) * 2.0 * lambda_;
            for (int j = 0; j < expertsNum_; ++j)
	    {
                for (size_t k = 0; k < cliqueIndexArrayInPixel[targetIndex].size(); ++k)
		{
                    int cliqueIndex = cliqueIndexArrayInPixel[targetIndex][k];
                    term_1 += pow(alpha_[j][getPixelIndexInClique(targetIndex, pixelIndexArrayInClique[cliqueIndex])], 2.0);
                }
            }
            
            double term_2 = 0.0;
            for (size_t j = 0; j < cliqueIndexArrayInPixel[targetIndex].size(); ++j)
	    {
                int cliqueIndex = cliqueIndexArrayInPixel[targetIndex][j];
                double term_3 = 0.0;
                for (int k = 0; k < expertsNum_; ++k)
		{
                    double term_4 = 0.0;
                    for (size_t l = 0; l < pixelIndexArrayInClique[cliqueIndex].size(); ++l)
		    {
                        int neighborIndex = pixelIndexArrayInClique[cliqueIndex][l];
                        if (neighborIndex != targetIndex)
			{
                            int pixelIndex = getPixelIndexInClique(neighborIndex, pixelIndexArrayInClique[cliqueIndex]);
                            term_4 += (alpha_[k][pixelIndex] * data[neighborIndex]);
                        }
                    }                    
                    term_3 += (alpha_[k][getPixelIndexInClique(targetIndex, pixelIndexArrayInClique[cliqueIndex])] * term_4);
                }                
                term_2 += (beta_[getPixelIndexInClique(targetIndex, pixelIndexArrayInClique[cliqueIndex])] - term_3);
            }
            
            double update = term_2 / term_1;
            double diff = update - data[targetIndex];
            distortion += (fabs(diff) / static_cast<double>(target.size()));
            data[targetIndex] += (omega * diff);
        }
        
        // 終了条件を確認
        if (distortion < epsilon)
	{
            break;
        }
        
        omega -= (omega0 / static_cast<double>(maxLoop));
        
        if (loop % 100 == 0)
	{
            std::cout << loop << ": " << distortion << std::endl;
        }        
    }
    
    return data;
}

void GaussFoE::loadParameter(const char *filename)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (ifs.is_open())
    {
        cleanUp();
        ifs.read(reinterpret_cast<char*>(&expertsNum_), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&cliqueWidth_), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&cliqueHeight_), sizeof(int));
        initialize();
        for (int i = 0; i < expertsNum_; ++i)
	{
            for (int j = 0; j < cliqueSize(); ++j)
	    {
                ifs.read(reinterpret_cast<char*>(&alpha_[i][j]), sizeof(double));
            }
        }
        
        for (int i = 0; i < cliqueSize(); ++i)
	{
            ifs.read(reinterpret_cast<char*>(&beta_[i]), sizeof(double));
        }
        
        ifs.read(reinterpret_cast<char*>(&lambda_), sizeof(double));
        ifs.close();
	
    }
    else
    {
        std::cerr << "Error: File " << filename << " can't open." << std::endl; 
    }
}

void GaussFoE::saveParameter(const char *filename)
{
    std::ofstream ofs(filename, std::ios::binary);
    if (ofs.is_open())
    {
        ofs.write(reinterpret_cast<char*>(&expertsNum_), sizeof(int));
        ofs.write(reinterpret_cast<char*>(&cliqueWidth_), sizeof(int));
        ofs.write(reinterpret_cast<char*>(&cliqueHeight_), sizeof(int));
        for (int i = 0; i < expertsNum_; ++i)
	{
            for (int j = 0; j < cliqueSize(); ++j)
	    {
                ofs.write(reinterpret_cast<char*>(&alpha_[i][j]), sizeof(double));
            }
        }
        
        for (int i = 0; i < cliqueSize(); ++i)
	{
            ofs.write(reinterpret_cast<char*>(&beta_[i]), sizeof(double));
        }
        
        ofs.write(reinterpret_cast<char*>(&lambda_), sizeof(double));
        ofs.close();
	
    }
    else
    {
        std::cerr << "Error: File " << filename << " can't open." << std::endl; 
    }
}

void GaussFoE::initialize()
{
    alpha_ = new double*[expertsNum_];
    for (int i = 0; i < expertsNum_; ++i)
    {
        alpha_[i] = new double[cliqueSize()];
        for (int j = 0; j < cliqueSize(); ++j)
	{
            alpha_[i][j] = 0.0;
        }
    }
    
    beta_ = new double[cliqueSize()];
    for (int i = 0; i < cliqueSize(); ++i)
    {
        beta_[i] = 0.0;
    }
}

void GaussFoE::cleanUp()
{
    if (alpha_ != NULL)
    {
        for (int i = 0; i < expertsNum_; ++i)
	{
            delete[] alpha_[i];
        }
        delete[] alpha_;
        alpha_ = NULL;
    }
    
    if (beta_ != NULL)
    {
        delete[] beta_;
        beta_ = NULL;
    }
}

// クリークのサイズから画素とクリークを対応付ける
void GaussFoE::defineClique(std::vector<std::vector<int> > &pixelIndexArrayInClique,
                            std::vector<std::vector<int> > &cliqueIndexArrayInPixel,
                            int width, int height)
{
    pixelIndexArrayInClique.reserve((width - cliqueWidth_ + 1) * (height - cliqueHeight_ + 1));
    cliqueIndexArrayInPixel.resize(width * height);
    for (int y = 0; y < height - cliqueHeight_ + 1; ++y)
    {
        for (int x = 0; x < width - cliqueWidth_ + 1; ++x)
	{
            std::vector<int> pixelIndexArray;
            for (int i = 0; i < cliqueHeight_; ++i)
	    {
                for (int j = 0; j < cliqueWidth_; ++j)
		{
                    int pixelIndex = (y + i) * width + x + j;
                    pixelIndexArray.push_back(pixelIndex);
                    int cliqueIndex = pixelIndexArrayInClique.size();
                    cliqueIndexArrayInPixel[pixelIndex].push_back(cliqueIndex);
                }
            }
            pixelIndexArrayInClique.push_back(pixelIndexArray);
        }
    }
}

int GaussFoE::getPixelIndexInClique(int pixelIndex, const std::vector<int> &pixelIndexArray)
{
    for (size_t i = 0; i < pixelIndexArray.size(); ++i)
    {
        if (pixelIndex == pixelIndexArray[i])
	{
            return i;
        }
    }
    
    std::cerr << "Not found PixelIndex in FoE::getPixelIndexInClique(" << pixelIndex << ")" << std::endl;
    return -1;
}

void inverseMatrix(double const * const * src, double **dst, int dim)
{
    for (int i = 0; i < dim; ++i)
    {
	for (int j = 0; j < dim; ++j)
	{
	    dst[i][j] = src[i][j];
	}
    }
	
    int *rows = new int[dim];
	
    for (int ipv = 0; ipv < dim; ++ipv)
    {
        double big = 0.0;
	int pivot_row;
	for (int i = ipv; i < dim; ++i)
	{
	    if (fabs(dst[i][ipv]) > big)
	    {
		big = fabs(dst[i][ipv]);
		pivot_row = i;
	    }
	}
	
        rows[ipv] = pivot_row;
	    
	if (ipv != pivot_row)
	{
	    for (int i = 0; i < dim; ++i)
	    {
		double temp = dst[ipv][i];
		dst[ipv][i] = dst[pivot_row][i];
		dst[pivot_row][i] = temp;
	    }
        }
	    
	double inv_pivot = 1.0 / dst[ipv][ipv];
	dst[ipv][ipv] = 1.0;
	
        for (int j = 0; j < dim; ++j)
	{
	    dst[ipv][j] *= inv_pivot;
	}
	    
	for (int i = 0; i < dim; i++)
	{
	    if (i != ipv)
	    {
		double temp = dst[i][ipv];
		dst[i][ipv] = 0.0;
		for (int j = 0; j < dim; ++j)
		{
		    dst[i][j] -= temp * dst[ipv][j];
		}
	    }
	}
    }
	
    for (int j = dim - 1; j >= 0; --j)
    {
	if (j != rows[j])
	{
	    for (int i = 0; i < dim; ++i)
	    {
		double temp = dst[i][j];
		dst[i][j] = dst[i][rows[j]];
		dst[i][rows[j]] = temp;
	    }
	}
    }
    
    delete[] rows; rows = NULL;
}
