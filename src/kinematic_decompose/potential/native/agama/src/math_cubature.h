/** \file   math_cubature.h
    \brief  N-dimensional adaptive-refinement integration
    \author Eugene Vasiliev
    \date   2025

    This module provides the routine for N-dimensional integration with adaptive refinement,
    based on the Genz-Malik (1980) rule.
    The approach follows the one from the Cubature library by S.Johnson,
    with the following additional features (partly inspired by Berntsen et al. 1991),
    aimed at making the method more reliable at the cost of a modest (~10-30%) increase
    in the number of function evaluations:
    - Two-level error estimate (use the difference between the integral over the parent cell
      and two child cells as another measure of error, in addition to the main error estimate
      from the difference between 7th and 5th-order rules in each cell).
    - The above method does not apply to the root cell, so we obtain the second error estimate
      as the difference between 7th and 3rd-order rules, ensuring that the root cell is refined
      unless both error estimates are negligible (e.g. the integrand is a low-order polynomial).
    - In the N=1 case, use a 11-point Gauss-Kronrod rule instead of 15-point, and also add another
      lower-order estimate of error in the root cell for the same reason.
    - Use compensated summation (aka Kahan) for the total integral and its error estimate,
      to avoid the loss of precision after repeated subtractions and additions of cell estimates.
    - In case that the splitting direction cannot be determined (e.g. because the integrand
      is zero inside most of the cell), cycle between all dimensions.
    - Use a generic number type for the integrand (can be adapted to work, e.g., with dual numbers).
    The class CubatureWorker takes an N-dimensional function that provides M>=1 values at each point,
    and computes M integrals simultaneously, trying to reach the required relative tolerance for
    each of these M components, but the implicit assumption is that they are of comparable magnitude
    (i.e., the error estimate in each cell is summed over all M components, and the cell with
    the largest absolute error is refined in each iteration, rather the the cell with the largest
    relative error in the most uncertain component).
*/
#pragma once
#include "math_base.h"
#include <queue>
#include <cmath>
#include <stdexcept>
#include <alloca.h>

namespace math {

// custom type for result accumulation
typedef double ResultType;

// make sure that abs(xxx) picks up std::abs for double, and the appropriate function for a custom ResultType
using std::abs;
using std::max;

/// Class that performs the integration using the run() method.
class CubatureWorker {
private:
    /** A base hypercube subregion in which the integral and its error estimate
        are computed using the Genz-Malik (or Gauss-Kronrod in 1d) cubature rule.
        This lightweight structure is used in the priority_queue container and contains
        very little data - essentially just the index in the global arrays containing
        cell coordinates and integral values, plus the preferred dimension for splitting.
        Instances of this structure can be ordered based on the error estimate.
    */
    struct Cell {
        unsigned int index; ///< index of this cell's coordinates, values and errors in the global arrays
        short splitDim;     ///< index of the dimension along which the cell should be split next time
        short level;        ///< number of binary splits needed to reach this cell
        ResultType errmax;  ///< cached value of maximum error among all integrands in this cell
        inline bool operator< (const Cell &other) const  ///< sort cells in the queue based on maxerr
        { return errmax < other.errmax; }
    };

    const IFunctionNdim& fnc;  ///< function to be integrated
    const int N;               ///< numVars - dimension of space
    const int M;               ///< numValues - independent integrands (function values / components)
    const int pointsPerCell;   ///< number of integration points in each cell (depends on N)
    const double relToler;     ///< required relative error in each component of the integral
    const int maxNumEval;      ///< upper limit on the number of function evaluations
    int numEval;               ///< current number of function evaluations

    /// additional fixed-size temporary storage space for the arrays listed below
    std::vector<ResultType> temp;

    /// extra bits of precision for the compensated accumulation of totalResult and totalError
    /// (M values for each; allocated within 'temp')
    ResultType* totalResultExtra;
    ResultType* totalErrorExtra;

    /// result (M integral values in one cell that was refined); allocated within 'temp'
    ResultType* resultPrevLevel;

    /// global values of all M integrals; points to an external storage (output of this routine)
    ResultType* totalResult;

    /// global error estimates of all M integrals; either external storage or allocated within 'temp'
    ResultType* totalError;

    /// pointer for storing the total number of function evaluations at the end (skipped if NULL)
    int* numEvalPtr;

    /// 2 N values for each cell: coordinates of the center followed by half box sizes
    std::vector<double> coords;

    /// 2 M values for each cell: components of the integral followed by their error estimates
    std::vector<ResultType> results;

    /// collection of all cells sorted by the error estimates
    std::priority_queue<Cell, std::vector<Cell> > queue;

    /// convenience functions for easier access to cell coordinates and integration results
    inline std::vector<double>::iterator center (const Cell& cell)
    { return coords.begin() + cell.index * 2 * N; }

    inline std::vector<double>::iterator halfbox(const Cell& cell)
    { return coords.begin() + cell.index * 2 * N + N; }

    inline std::vector<ResultType>::iterator result(const Cell& cell)
    { return results.begin() + cell.index * 2 * M; }

    inline std::vector<ResultType>::iterator error(const Cell& cell)
    { return results.begin() + cell.index * 2 * M + M; }

    /// assign coordinates for integration points in one cell, using the 1d Gauss-Kronrod rule
    void preparePointCoordsGK(const Cell& cell, double* points)
    {
        const double xgk[5] = {
            0.9840853600948425,
            0.9061798459386640,
            0.7541667265708493,
            0.5384693101056831,
            0.2796304131617832
        };
        const double center1 = center(cell)[0], boxsize1 = halfbox(cell)[0];
        for(unsigned int i=0; i<5; i++) {
            points[i]    = center1 - boxsize1 * xgk[i];
            points[10-i] = center1 + boxsize1 * xgk[i];
        }
        points[5] = center1;
    }

    /// compute the integral and its error estimate in one cell, using the 1d Gauss-Kronrod rule
    void computeResultsGK(Cell& cell, const double* fncvalues)
    {
        const double wg[3] = {  // weights of the 5-point Gauss rule, 9th order estimate
            0.2369268850561891,
            0.4786286704993664,
            0.5688888888888889
        };
        const double wk[6] = {  // weights of the 11-point Kronrod rule
            0.04258203675108176,
            0.1152333166224734,
            0.1868007965564926,
            0.2410403392286475,
            0.2728498019125589,
            0.2829874178574912
        };
        const double wx[5] = {  // weights of a custom 8-point rule, 7th order estimate
            0.082576000557218330026,
            0,
            0.36614781191987978023,
            0.0094720491899567308709,
            0.54180413833294515887
        };
        const double boxsize = halfbox(cell)[0];
        cell.errmax = 0;
        for(int v=0; v<M; v++) {
            ResultType
                resultGauss   = fncvalues[5 * M + v] * wg[2],
                resultKronrod = fncvalues[5 * M + v] * wk[5],
                resultExtra   = 0;
            for(unsigned int i=0; i<5; i++) {
                ResultType fv = (fncvalues[i * M + v] + fncvalues[(10-i) * M + v]);
                resultKronrod  += fv * wk[i];
                resultExtra    += fv * wx[i];
                if(i%2==1)
                    resultGauss+= fv * wg[i/2];
            }
            // use two error estimates of a different order for greater fidelity
            ResultType err = max(abs(resultGauss - resultKronrod), abs(resultExtra - resultKronrod));
            result(cell)[v] = resultKronrod * boxsize;
            error (cell)[v] = err * boxsize;
            cell.errmax = max(cell.errmax, err * boxsize);
        }
        cell.splitDim = 0;
    }

    /// assign coordinates for integration points in one cell in the case N>1, using the Genz-Malik rule
    void preparePointCoordsGM(const Cell& cell, double* points)
    {
        const double
            lambda2 = sqrt(9./70),
            lambda3 = sqrt(9./10),
            lambda4 = lambda3,
            lambda5 = sqrt(9./19);
        const std::vector<double>::const_iterator begin = center(cell);
        const std::vector<double>::const_iterator boxsize = halfbox(cell);
        // 0th point is at the center of the box
        points = std::copy(begin, begin + N, points);  // copy and advance the "points" pointer by N
        // next 4N points are along the principal axes
        for(int d=0; d<N; d++) {
            points = std::copy(begin, begin + N, points);
            points[d-N] -= boxsize[d] * lambda2;
            points = std::copy(begin, begin + N, points);
            points[d-N] += boxsize[d] * lambda2;
            points = std::copy(begin, begin + N, points);
            points[d-N] -= boxsize[d] * lambda3;
            points = std::copy(begin, begin + N, points);
            points[d-N] += boxsize[d] * lambda3;
        }
        // next 2N(N-1) points are in the corners of each principal plane
        for(int d1=0; d1<N-1; d1++) {
            for(int d2=d1+1; d2<N; d2++) {
                points = std::copy(begin, begin + N, points);
                points[d1-N] -= boxsize[d1] * lambda4;
                points[d2-N] -= boxsize[d2] * lambda4;
                points = std::copy(begin, begin + N, points);
                points[d1-N] -= boxsize[d1] * lambda4;
                points[d2-N] += boxsize[d2] * lambda4;
                points = std::copy(begin, begin + N, points);
                points[d1-N] += boxsize[d1] * lambda4;
                points[d2-N] -= boxsize[d2] * lambda4;
                points = std::copy(begin, begin + N, points);
                points[d1-N] += boxsize[d1] * lambda4;
                points[d2-N] += boxsize[d2] * lambda4;
            }
        }
        // finally, 2^N points are in the corners of the hypercube
        for(int dd=0; dd<(1<<N); dd++) {
            points = std::copy(begin, begin + N, points);
            for(int d=0; d<N; d++)
                points[d-N] += boxsize[d] * ((dd & (1<<d)) ? +1 : -1) * lambda5;
        }
    }

    /// compute the integral and its error estimate in one cell, using the Genz-Malik rule for N>1
    void computeResultsGM(Cell& cell, const double* fncvalues)
    {
        const double
            // weights for the high(7)-order rule;  this magic number 19683 is 3**9
            weighth1 = (1.  / 19683) * (12824 - N * (9120 - 400 * N)),     // group 1: central point
            weighth2 = 2940./ 19683,           // groups 2 and 3: points along each principal axis
            weighth3 = (1.  / 19683) * (1820 - 400 * N),
            weighth4 = 200. / 19683,           // group 4: corners of each principal plane
            weighth5 = 6859./ 19683 / (1<<N),  // group 5: all corners of the hypercube
            // weights for the low(5)-order rule
            weightl1 = 1 - N * (19 - N) * (50./729),
            weightl2 = 245./ 486,
            weightl3 = (1. / 1458) * (265 - 100 * N),
            weightl4 = 25. / 729;

        double vol = 1<<N;
        for(int d=0; d<N; d++)
            vol *= halfbox(cell)[d];

        // finite-difference estimate of the fourth derivative along each dimension,
        // and an auxiliary array of the same size tracking the floating-point roundoff error
        ResultType* diff = static_cast<ResultType*>(alloca(2 * N * sizeof(ResultType)));
        ResultType* roff = diff + N;
        std::fill(diff, diff + 2 * N, 0);

        cell.errmax = 0;
        for(int v=0; v<M; v++) {
            ResultType val1 = fncvalues[v], sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0;
            // groups 2 and 3: 4N points along the principal axes (+-lambda2, +-lambda3)
            for(int d=0; d<N; d++) {
                ResultType
                val2 = fncvalues[(d*4+1) * M + v] + fncvalues[(d*4+2) * M + v],
                val3 = fncvalues[(d*4+3) * M + v] + fncvalues[(d*4+4) * M + v];
                diff[d] += abs(val2 - 2*val1 - (1./7) * (val3 - 2*val1));
                roff[d] += abs(val2) + (16./7) * abs(val1) + (1./7) * abs(val3);
                sum2 += val2;
                sum3 += val3;
            }
            // group 4: 2N(N-1) points in the corners of each principal plane (lambda4)
            for(int dd=0; dd<2*N*(N-1); dd++)
                sum4 += fncvalues[(N*4+1 + dd) * M + v];
            // group 5: 2^N points in the corners of the hypercube (lambda5)
            for(int dd=0; dd<(1<<N); dd++)
                sum5 += fncvalues[(2*N*(N+1)+1 + dd) * M + v];

            // compute results with low(5th) and high(7th) order rules
            ResultType resultl = vol *
                (weightl1 * val1 + weightl2 * sum2 + weightl3 * sum3 + weightl4 * sum4);
            ResultType resulth = vol *
                (weighth1 * val1 + weighth2 * sum2 + weighth3 * sum3 + weighth4 * sum4 + weighth5 * sum5);
            ResultType err = abs(resulth - resultl);

            // extra precaution for the root cell only:
            // compute another integral estimate using a 3rd-order rule, and a corresponding
            // error estimate as the difference between 7th and 3rd-order rules; then take
            // the higher of the two estimates (7-5 and 7-3) to prevent a rare but catastrophic
            // situation when the 5-th and 7-th order rules happen to be close to each other,
            // but both being far from truth (so the cell is mistakenly not refined any further).
            // This precaution would be less useful at later stages, because the two-level error
            // estimate also guards against an accidental underestimate of actual error.
            if(cell.level==0) {
                const double
                    weightx1 = 1 - 10./27 * N,
                    weightx3 = 5./27;
                ResultType resultx = vol * (weightx1 * val1 + weightx3 * sum3);
                err = max(abs(resulth - resultx) * 0.2 /*fudge factor*/, err);
            }

            result(cell)[v] = resulth;
            error (cell)[v] = err;
            cell.errmax = max(cell.errmax, err);
        }

        // find out which dimension to split: first find out the largest fourth difference
        ResultType maxDiff(0);
        for(int d=0; d<N; d++) {
            if(diff[d] > maxDiff) {
                maxDiff = diff[d];
            }
        }
        // next collect the list of dimensions with similar differences as candidates for splitting;
        // comparison takes into account roundoff errors on each value
        short* splitDims = static_cast<short*>(alloca(N * sizeof(short)));
        short countDims = 0;
        for(int d=0; d<N; d++) {
            if(diff[d] + roff[d] * DBL_EPSILON >= maxDiff)
                splitDims[countDims++] = d;
        }
        if(countDims == 0)  // guard against pathological cases (?)
            countDims = N;

        // split along a dimension from the list of candidate dimensions with similar differences,
        // alternating the order on each level in case of ties (this helps to overcome
        // semi-pathological cases, e.g. when the differences are all zero, or result is unreliable)
        cell.splitDim = splitDims[cell.level % countDims];
    }

    /// accumulate the result with extra precision (Kahan-style)
    static void addCompensated(const ResultType& value, ResultType& accum, ResultType& extra)
    {
        ResultType tmp = value + accum;
        extra += abs(value) > abs(accum) ? (value - tmp) + accum : (accum - tmp) + value;
        accum = tmp;
    }

    /// remove the current top cell from the queue in preparation for its splitting,
    /// and update the global estimates of integral values and errors
    Cell popCellFromQueue()
    {
        Cell cell = queue.top();
        // save the estimates of integrals in the temporary storage
        // for subsequent use in the two-level error estimate
        std::copy(result(cell), result(cell) + M, resultPrevLevel);
        // subtract this cell's contribution to the total results
        for(int v=0; v<M; v++) {
            addCompensated(-result(cell)[v], totalResult[v], totalResultExtra[v]);
            addCompensated(-error (cell)[v], totalError [v], totalErrorExtra [v]);
        }
        // wipe out the integrals & errors for this cell from the overall array
        std::fill(result(cell), result(cell) + 2*max(0,M), 0);
        // check for possible loss of precision in the total results & errors,
        // and recompute them from scratch, if needed
        for(int v=0; v<M; v++) {
            if(abs(totalResultExtra[v]) > abs(totalResult[v]) * relToler) {
                totalResult[v] = 0;
                totalResultExtra[v] = 0;
                for(size_t i=v; i<results.size(); i+=2*M)
                    addCompensated(results[i], totalResult[v], totalResultExtra[v]);
            }
            if(abs(totalErrorExtra[v]) > abs(totalError[v]) * relToler) {
                totalError[v] = 0;
                totalErrorExtra[v] = 0;
                for(size_t i=v+M; i<results.size(); i+=2*M)
                    addCompensated(results[i], totalError[v], totalErrorExtra[v]);
            }
        }
        // remove cell from the queue, and reorder the remaining cells as needed
        queue.pop();
        return cell;
    }

    /** Split the given cell into two halves along the pre-computed dimension.
        \param[out] cell1  is the cell to split, which is taken (and removed) from the queue
        and replaced by one of its two children (without putting it back yet).
        \param[out] cell2  is the other (newly created) child cell.
        Both children cells should not be put back into the queue until they have been processed
        (results and error estimates are obtained).
    */
    void splitCell(Cell cells[2])
    {
        cells[0] = popCellFromQueue();
        // allocate the room for the new cell (one of the two children)
        cells[1].index = coords.size() / (2 * N);  // = the current number of cells
        coords .resize(  coords.size() +  2 * N);  // arrays are extended to accommodate the new cell
        results.resize( results.size() +  2 * M);
        // copy the coordinates (center and width) from the first cell into the second one
        std::copy(center(cells[0]), center(cells[0]) + 2 * N, center(cells[1]));
        // adjust the center location in the given dimension in both cells
        double newboxsize = halfbox(cells[0])[cells[0].splitDim] * 0.5;
        center (cells[0])[cells[0].splitDim] -= newboxsize;
        center (cells[1])[cells[0].splitDim] += newboxsize;
        // adjust the box size in the given dimension in both cells
        halfbox(cells[0])[cells[0].splitDim] = newboxsize;
        halfbox(cells[1])[cells[0].splitDim] = newboxsize;
        cells[1].level = ++cells[0].level;  // keep track of the refinement level
        // the below is not necessary; splitDim will be overwritten immediately by computeResults**
        cells[1].splitDim = cells[0].splitDim = -1;
    }

    /** Process one or two cells:
        - prepare integration points in all cells according to the chosen rule;
        - call the user-provided function to obtain the integrand values at these points;
        - compute the integral estimate and its error;
        - (optional) correct the error estimate using the two-level scheme;
        - add back the results to the total tally.
    */
    void processCells(unsigned int numCells, Cell cells[])
    {
        if(numCells == 0)  // never happens, but silences an unjustified compiler warning
            return;
        double* points = static_cast<double*>(alloca(
            numCells * pointsPerCell * N * sizeof(double)));
        ResultType* fncvalues = static_cast<double*>(alloca(
            numCells * pointsPerCell * M * sizeof(ResultType)));  // assume that M is not too large!

        for(unsigned int c=0; c<numCells; c++) {
            if(N==1)
                preparePointCoordsGK(cells[c], points + c * pointsPerCell);
            else
                preparePointCoordsGM(cells[c], points + c * pointsPerCell * N);
        }

        // evaluate the function for all points in these cells at once
        fnc.evalMany(numCells * pointsPerCell, points, fncvalues);

        // compute the integrals and error estimates in each cell
        for(unsigned int c=0; c<numCells; c++) {
            if(N==1)
                computeResultsGK(cells[c], fncvalues + c * pointsPerCell * M);
            else
                computeResultsGM(cells[c], fncvalues + c * pointsPerCell * M);
        }

        // assign a more conservative error estimate for both cells, comparing the integral values
        // over both cells with the value of the single parent cell at the previous level
        // (does not apply on the first call, when there is only one root cell)
        if(numCells>1) {
            for(int v=0; v<M; v++) {
                ResultType sumerr(0);
                for(unsigned int c=0; c<numCells; c++) {
                    resultPrevLevel[v] -= result(cells[c])[v];
                    sumerr += error(cells[c])[v];
                }
                ResultType mult = abs(resultPrevLevel[v]);
                for(unsigned int c=0; c<numCells; c++) {
                    // coefficients in eq.12 of Berntsen+91; lower than their fiducial values
                    const double C5 = 0.4, C6 = 0.1;
                    ResultType ratio = sumerr==0 ? 1 : error(cells[c])[v] / sumerr;
                    ResultType add = (C5 * ratio + C6) * mult;
                    ResultType err = add + error(cells[c])[v];
                    error(cells[c])[v] = err;
                    cells[c].errmax = max(cells[c].errmax, err);
                }
            }
        }

        // add the results & errors of the new cells to the overall tally
        for(unsigned int c=0; c<numCells; c++) {
            for(int v=0; v<M; v++) {
                addCompensated(result(cells[c])[v], totalResult[v], totalResultExtra[v]);
                addCompensated(error (cells[c])[v], totalError [v], totalErrorExtra [v]);
            }
        }

        // add cells back to the queue, keeping it sorted according to error estimates
        for(unsigned int c=0; c<numCells; c++)
            queue.push(cells[c]);
        numEval += numCells * pointsPerCell;
    }

public:
    /** Prepare ground for integration, which is then performed by the run() method.
        \param[in]  fnc  is the N-dimensional function returning M>=1 values.
        \param[in]  xlower, xupper  are two opposite corners of the integration box (each length N).
        \param[in] _relToler  is the required relative error in each of the M integrals.
        \param[in] _maxNumEval  is the upper limit on the number of function evaluations: integration
        will stop once this number is exceeded, even if it did not reach the required tolerance.
        Note that the input function is called in a vectorized way for batches of many points at once,
        but the number of evaluations counts the total number of points, not the number of calls.
        \param[out] _result  will contain the integrals (should be a preallocated array of length M).
        \param[out] _error  if not NULL, will contain the corresponding error estimates.
        \param[out] _numEval  if not NULL, will store the total number of function evaluations.
    */
    CubatureWorker(const IFunctionNdim& _fnc,
        const double xlower[], const double xupper[],
        const double _relToler,
        const int _maxNumEval,
        ResultType _result[],
        ResultType _error[] = NULL,
        int* _numEval = NULL)
    :
        fnc(_fnc),
        N(fnc.numVars()),
        M(fnc.numValues()),
        pointsPerCell(N==1 ? 11 : 1 + 2 * N * (N+1) + (1<<N)),
        relToler(_relToler),
        maxNumEval(_maxNumEval),
        numEval(0),
        temp(_error==NULL ? 4*M : 3*M),
        totalResultExtra(&temp[0]),
        totalErrorExtra (&temp[M]),
        resultPrevLevel (&temp[M*2]),
        totalResult(_result),
        totalError (_error==NULL ? &temp[3*M] : _error),
        numEvalPtr(_numEval),
        coords (2*N),  // allocate the space for the coordinates of the root cell
        results(2*M)   // and the same for the results
    {
        if(N<1)
            throw std::runtime_error("integrateNdim: number of dimensions must be positive");
        if(N>10)
            throw std::runtime_error("integrateNdim: number of dimensions is too large");
        if(M<1)
            throw std::runtime_error("integrateNdim: number of function values must be at least one");
        if(M>10000)
            throw std::runtime_error("integrateNdim: number of function values is too large");
        // store the coordinates for the root cell
        for(int d=0; d<N; d++) {
            coords[d  ] = (xupper[d] + xlower[d]) * 0.5;  // center
            coords[d+N] = (xupper[d] - xlower[d]) * 0.5;  // half-width
        }
    }

    /** Main loop, repeatedly splitting the cell with the highest error and processing
        the two child cells, until the error estimate drops below the threshold or 
        the total number of function evaluations exceeds the limit.
    */
    void run()
    {
        std::fill(totalResult, totalResult + M, 0);
        std::fill(totalError,  totalError  + M, 0);
        // create the root cell
        Cell rootCell;
        rootCell.index = 0;
        rootCell.level = 0;
        rootCell.splitDim = -1;
        processCells(1, &rootCell);
        while(numEval < maxNumEval) {
            // compute the integrand and its error estimate
            bool converged = true;
            for(int v=0; v<M; v++)
                converged &= totalError[v] <= abs(totalResult[v]) * relToler || !isFinite(totalResult[v]);
            if(converged)
                break;
            // take the first cell in the queue and split it into two child cells
            Cell cells[2];
            splitCell(cells);
            processCells(2, cells);
        }
        // finally, add the accumulated extra bits of precision to the total result & error
        for(int v=0; v<M; v++) {
            totalResult[v] += totalResultExtra[v];
            totalError [v] += totalErrorExtra [v];
        }
        if(numEvalPtr)
            *numEvalPtr = numEval;
    }
};

}  // namespace
