#ifndef _POTENTIAL_HPP
#define _POTENTIAL_HPP

#include <cstdlib>
#include <cmath>
#include <complex>
#include <algorithm>
#include <string>
#include "misc/crasher.hpp"
#include "misc/randomer.hpp"
#include "misc/vector.hpp"
#include "misc/matrixop.hpp"
#include "misc/ioer.hpp"

namespace {
    using std::vector;
    using std::complex;

    double param_A = 0.0025;
    double param_B = 0.01;
    double param_k = 0.8;
    double param_W = 0.5;

    void output_potential_param() {
        /*
         * output potential paramters
         */
        ioer::info("# Potential parameters: ", 
                    " A = ", param_A,
                    " B = ", param_B,
                    " k = ", param_k,
                    " W = ", param_W,
                    ""
                    );
    }

    void set_potenial_params(const std::vector<double>& params) {
        misc::crasher::confirm(params.size() >= 4, 
                "set_potenial_params: potential paramter vector size must be >= 4");
        param_A = params[0];
        param_B = params[1];
        param_k = params[2];
        param_W = params[3];
    }

    complex<double> cal_phi(const vector<double>& r) {
        /*
         * helper
         */
        return param_W * r[1];
    }

    vector< complex<double> > cal_nablaphi(const vector<double>& r) {
        /*
         * helper
         */
        vector< complex<double> > nablaphi(r.size(), 0.0);
        nablaphi[1] = param_W;
        return nablaphi;
    }

    vector< complex<double> > cal_H(const vector<double>& r) {
        /*
         * input: position vector r
         *
         * return: complex Hamiltonian H(r)
         *
         */
        const double x = r[0];
        const double y = r[1];
        const double R= sqrt(x * x + y * y);
        double coef;
        const complex<double> eip = exp(matrixop::IMAGIZ * cal_phi(r));

        vector< complex<double> > H(4, 0.0);
        if (R > 0.0) {
            const double R2 = R * R;
            const double exp_R2 = exp(-R2);
            const double expkR= exp(param_k * R);
            coef = param_A * exp_R2 + param_B / R * (expkR - 1) / (expkR + 1);  
            H[0+0*2] = -x;
            H[1+1*2] = x;
            H[0+1*2] = y * eip;
            H[1+0*2] = conj(H[0+1*2]);
            H *= coef;
        }

        return H;
    }

    vector< vector< complex<double> > > cal_nablaH(const vector<double>& r) {
        /*
         * input: position vector r
         *
         * return: gradiant of complex Hamiltonian (Hx, Hy, ...)
         *
         */
        const int ndim = r.size();

        const double x = r[0];
        const double y = r[1];
        const double R= sqrt(x * x + y * y);
        const complex<double> eip = exp(matrixop::IMAGIZ * cal_phi(r));
        const vector< complex<double> > nablaphi = cal_nablaphi(r);

        // initialize nablaH
        vector< vector< complex<double> > > nablaH(ndim);
        for (int ix(0); ix < ndim; ++ix) {
            nablaH[ix].resize(4, 0.0);
        }

        if (R > 0.0) {
            const double R2 = R * R;
            const double R3 = R2 * R;

            const double exp_R2 = exp(-R2);
            const double expkR= exp(param_k * R);

            const double part1 = x * exp_R2;
            const double part2 = x / R;
            const double part3 = x / (R * (expkR + 1));
            const double part4 = y * exp_R2;
            const double part5 = y / R;
            const double part6 = y / (R * (expkR + 1));

            const double dxdpart1 = exp_R2 * (1 - 2 * x * x);
            const double dydpart1 = -2 * x * y * exp_R2;

            const double dxdpart2 = y * y / R3;
            const double dydpart2 = -x * y / R3;

            const double dxdpart3 = (y * y / R * (expkR + 1) - param_k * x * x * expkR) / (R2 * pow(expkR + 1, 2));
            const double dydpart3 = (-x * y / R * (expkR + 1) - param_k * x * y * expkR) / (R2 * pow(expkR + 1, 2));

            const double dxdpart4 = dydpart1;
            const double dydpart4 = exp_R2 * (1 - 2 * y * y);

            const double dxdpart5 = dydpart2;
            const double dydpart5 = x * x / R3;

            const double dxdpart6 = dydpart3;
            const double dydpart6 = (x * x / R * (expkR + 1) - param_k * y * y * expkR) / (R2 * pow(expkR + 1, 2));


            // H11 = A * part1 + B * part2 - 2 * B * part3
            // H01 = (A * part4 + B * part5 - 2 * B * part6) * eip
            vector< complex<double> >& Hx = nablaH[0];
            vector< complex<double> >& Hy = nablaH[1];

            Hx.resize(4);
            Hx[1+1*2] = param_A * dxdpart1 + param_B * dxdpart2 - 2 * param_B * dxdpart3;
            Hx[0+0*2] = -Hx[1+1*2];
            Hx[0+1*2] = eip * (param_A * dxdpart4 + param_B * dxdpart5 - 2 * param_B * dxdpart6 
                                + matrixop::IMAGIZ * nablaphi[0] * (param_A * part4 + param_B * part5 - 2 * param_B * part6));
            Hx[1+0*2] = conj(Hx[0+1*2]);

            Hy.resize(4);
            Hy[1+1*2] = param_A * dydpart1 + param_B * dydpart2 - 2 * param_B * dydpart3;
            Hy[0+0*2] = -Hy[1+1*2];
            Hy[0+1*2] = eip * (param_A * dydpart4 + param_B * dydpart5 - 2 * param_B * dydpart6 
                                + matrixop::IMAGIZ * nablaphi[0] * (param_A * part4 + param_B * part5 - 2 * param_B * part6));
            Hy[1+0*2] = conj(Hy[0+1*2]);
        }

        return nablaH;
    }

    void cal_info_nume(const vector<double>& r,
            vector<double>& eva, 
            vector< vector< complex<double> > >& dc,
            vector< vector< complex<double> > >& F,
            vector< complex<double> >& lastevt)
    {
        /*
         * input:   position vector r
         *
         * output:  eigenvalues
         *          (phase aligned) derivative coupling matrix
         *          force matrix
         *
         * in/out:  last step eigenvectors lastevt (can be a empty vector)
         *              on exit, lastevt is replaced by current step evt
         *
         * -- calculation is performed in the numerical way
         */

        const vector< complex<double> > H = cal_H(r);
        const int ndim = r.size();
        const int edim = static_cast<int>(std::sqrt(static_cast<double>(H.size())));
        vector< complex<double> > evt;
        matrixop::hdiag(cal_H(r), eva, evt);

        // correct phase
        if (not lastevt.empty()) {
            auto tmp = matrixop::matCmat(lastevt, evt, edim);
            for (int j = 0; j < edim; ++j) {
                complex<double> eip = tmp[j+j*edim] / abs(tmp[j+j*edim]);
                for (int k = 0; k < edim; ++k) {
                    evt[k+j*edim] /= eip;
                }
            }
        }

        // F, dc
        const vector< vector< complex<double> > > nablaH = cal_nablaH(r);
        dc.resize(ndim);
        F.resize(ndim);

        for (int ix = 0; ix < ndim; ++ix) {
            dc[ix] = matrixop::matCmatmat(evt, nablaH[ix], evt, edim, edim);
            F[ix].assign(edim * edim, 0.0);

            for (int j = 0; j < edim; ++j) {
                for (int k = 0; k < edim; ++k) {
                    F[ix][j+k*edim] = -dc[ix][j+k*edim];
                    if (j == k) {
                        dc[ix][j+k*edim] = 0.0;
                    }
                    else {
                        dc[ix][j+k*edim] /= (eva[k] - eva[j]);
                    }
                }
            }
        }

        // save evt to lastevt
        lastevt = std::move(evt);
    }
};

#endif
