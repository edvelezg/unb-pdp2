// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: 3572$
// $Date: 2007-11-19 13:58:06 +0000 (Mon, 19 Nov 2007) $
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// ------------------------------------------------------------- 
#ifndef __CUDPP_MANAGER_H__
#define __CUDPP_MANAGER_H__

/** @brief Internal manager class for CUDPPP resources
  * 
  */
class CUDPPManager
{
public:

    CUDPPManager();
    ~CUDPPManager();
   
    //! @internal Convert an opaque handle to a pointer to a manager
    //! @param [in] cudppHandle Handle to the Manager object
    static CUDPPManager* getManagerFromHandle(CUDPPHandle cudppHandle)
    {
        return reinterpret_cast<CUDPPManager*>(cudppHandle);
    }

    //! @internal Get an opaque handle for this manager
    CUDPPHandle getHandle()
    {
        return reinterpret_cast<CUDPPHandle>(this);
    }
};

#endif // __CUDPP_PLAN_MANAGER_H__
