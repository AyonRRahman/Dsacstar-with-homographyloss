/*
Based on the DSAC++ and ESAC code.
https://github.com/vislearn/LessMore
https://github.com/vislearn/esac

Copyright (c) 2016, TU Dresden
Copyright (c) 2020, Heidelberg University
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the TU Dresden, Heidelberg University nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TU DRESDEN OR HEIDELBERG UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#define MAXLOSS 10000000.0 // clamp for stability

namespace dsacstar
{
	/**
	 * @brief Calculates the rotational distance in degree between two transformations.
	 * Translation will be ignored.
	 *
	 * @param trans1 Transformation 1.
	 * @param trans2 Transformation 2.
	 * @return Angle in degree.
	 */
	cv::Mat_<double> eye()
	{
		cv::Mat_<double> m = cv::Mat_<double>::zeros(3, 3); 

		m(0,0) = 1 ; 
		m(1,1) = 1 ;
		m(2,2) = 1 ;


		return m ;
	}
    
	double calcAngularDistance(const dsacstar::trans_t& trans1, const dsacstar::trans_t& trans2)
	{
		cv::Mat rot1 = trans1.colRange(0, 3).rowRange(0, 3);
		cv::Mat rot2 = trans2.colRange(0, 3).rowRange(0, 3);

		cv::Mat rotDiff= rot2 * rot1.t();
		double trace = cv::trace(rotDiff)[0];

		trace = std::min(3.0, std::max(-1.0, trace));
		return 180*acos((trace-1.0)/2.0)/PI;
	}

	// /**
	//  * @brief Weighted average of  translational error and rotational error between two pose hypothesis.
	//  * @param h1 Pose 1.
	//  * @param h2 Pose 2.
	//  * @param wRot Weight of rotation error.
	//  * @param wTrans Weight of translation error.
	//  * @param cut Apply soft clamping after this value.
	//  * @return Loss.
	//  */
	// // double loss(
	// 	const dsacstar::trans_t& trans1, 
	// 	const dsacstar::trans_t& trans2, 
	// 	double wRot = 1.0, 
	// 	double wTrans = 1.0,
	// 	double cut = 100)
	// {
	// 	double rotErr = dsacstar::calcAngularDistance(trans1, trans2);
	// 	double tErr = cv::norm(
	// 		trans1.col(3).rowRange(0, 3) - trans2.col(3).rowRange(0, 3));

	// 	double loss = wRot * rotErr + wTrans * tErr;

	// 	if(loss > cut)
	// 		loss = std::sqrt(cut * loss);

	//     return std::min(loss, MAXLOSS);
	// }
	/**
	 * @brief Weighted average of  translational error and rotational error between two pose hypothesis.
	 * @param h1 Pose 1.
	 * @param h2 Pose 2.
	 * @param wRot Weight of rotation error.
	 * @param wTrans Weight of translation error.
	 * @param cut Apply soft clamping after this value.
	 * @return Loss.
	 */
	double loss(
		const dsacstar::trans_t& trans1, 
		const dsacstar::trans_t& trans2, 
		double xmin ,
		double xmax ,
		double cut = 100 )
	{
		// please note that trans1 is the estimated pose   est (4x4) 
		//       			trans2 is the ground truth pose gt (4x4)
		
		cv::Mat_<double> c_n = cv::Mat_<double>::zeros(3,1) ; 
		c_n(2,0) = -1 ; 

		cv::Mat_<double> w_R_c, w_R_chat, chat_R_c;
		
		//double rotErr = dsacstar::calcAngularDistance(trans1, trans2);

		// getting the rotation matrices out of the transformation 
		w_R_c    = trans2.colRange(0,3).rowRange(0, 3) ;       // gt 
		w_R_chat = trans1.colRange(0,3).rowRange(0, 3) ;       // est 

		chat_R_c = (w_R_chat.t()) * w_R_c ;  

		// getting the translationa terms 


		cv::Mat_<double> chat_t_c = trans2.col(3).rowRange(0, 3) - trans1.col(3).rowRange(0, 3);
		chat_t_c = w_R_chat.t() * chat_t_c ;
		
		// calculating the loss
		cv::Mat_<double> I = dsacstar::eye() ;  
		cv::Mat_<double> A = (I - chat_R_c ) * ((I- chat_R_c).t()) ;
		cv::Mat_<double> B = (c_n * (chat_t_c.t()) *(I - chat_R_c)) + ((c_n * (chat_t_c.t()) *(I - chat_R_c)).t()) ; 
		cv::Mat_<double> C = (c_n * (chat_t_c.t())) * ((c_n * (chat_t_c.t())).t()) ; 
        // cout<<"A"<<A<<endl;
        // cout<<"B"<<B<<endl;
        // cout<<"C"<<C<<endl;

		cv::Mat_<double> L = A + ((B /(xmax - xmin) ) *  log(xmax/xmin)) + (C / (xmin* xmax)) ;   
		// cout<<"L"<<L<<endl;

		double loss  = cv::trace(L)[0] ; 
		// cout<<"L"<<loss<<endl;
        if(loss > cut)
			loss = std::sqrt(cut * loss);

	    return std::min(loss, MAXLOSS);
	}


	// /**
	//  * @brief Calculate the derivative of the loss w.r.t. the estimated pose.
	//  * @param est Estimated pose (6 DoF).
	//  * @param gt Ground truth pose (6 DoF).
	//  * @param wRot Weight of rotation error.
	//  * @param wTrans Weight of translation error.
	//  * @param cut Apply soft clamping after this value.	 
	//  * @return 1x6 Jacobean.
	//  */
	// cv::Mat_<double> dLoss(
	// 	const dsacstar::pose_t& est, 
	// 	const dsacstar::pose_t& gt,
	// 	double wRot = 1.0,
	// 	double wTrans = 1.0,
	// 	double cut = 100)
	// {
	//     cv::Mat rot1, rot2, dRod;
	//     cv::Rodrigues(est.first, rot1, dRod);
	//     cv::Rodrigues(gt.first, rot2);

	//     // measure loss of inverted poses (camera pose instead of scene pose)
	//     cv::Mat_<double> invRot1 = rot1.t();
	//     cv::Mat_<double> invRot2 = rot2.t();

	//     // get the difference rotation
	//     cv::Mat diffRot = rot1 * invRot2;

	//     // calculate rotational and translational error
	//     double trace = cv::trace(diffRot)[0];
	//     trace = std::min(3.0, std::max(-1.0, trace));
	//     double rotErr = 180*acos((trace-1.0)/2.0)/CV_PI;

	//     cv::Mat_<double> invT1 = est.second.clone();
	//     invT1 = invRot1 * invT1;

	//     cv::Mat_<double> invT2 = gt.second.clone();
	//     invT2 = invRot2 * invT2;

	//     // zero error, abort
	//     double tErr = cv::norm(invT1 - invT2);

	//     cv::Mat_<double> jacobean = cv::Mat_<double>::zeros(1, 6);
	    
	//     // clamped loss, return zero gradient if loss is bigger than threshold
	//     double loss = wRot * rotErr + wTrans * tErr;
	//     bool cutLoss = false;


	//     if(loss > cut)
	//     {
	//     	loss = std::sqrt(loss);
	//     	cutLoss = true;
	//     }

	//     if(loss > MAXLOSS)
	//         return jacobean;

	//     if((tErr + rotErr) < EPS)
	//         return jacobean;
		
	  	
	//     // return gradient of translational error
	//     cv::Mat_<double> dDist_dInvT1(1, 3);
	//     for(unsigned i = 0; i < 3; i++)
	//         dDist_dInvT1(0, i) = (invT1(i, 0) - invT2(i, 0)) / tErr;

	//     cv::Mat_<double> dInvT1_dEstT(3, 3);
	//     dInvT1_dEstT = invRot1;

	//     cv::Mat_<double> dDist_dEstT = dDist_dInvT1 * dInvT1_dEstT;
	//     jacobean.colRange(3, 6) += dDist_dEstT * wTrans;

	//     cv::Mat_<double> dInvT1_dInvRot1 = cv::Mat_<double>::zeros(3, 9);

	//     dInvT1_dInvRot1(0, 0) = est.second.at<double>(0, 0);
	//     dInvT1_dInvRot1(0, 3) = est.second.at<double>(1, 0);
	//     dInvT1_dInvRot1(0, 6) = est.second.at<double>(2, 0);

	//     dInvT1_dInvRot1(1, 1) = est.second.at<double>(0, 0);
	//     dInvT1_dInvRot1(1, 4) = est.second.at<double>(1, 0);
	//     dInvT1_dInvRot1(1, 7) = est.second.at<double>(2, 0);

	//     dInvT1_dInvRot1(2, 2) = est.second.at<double>(0, 0);
	//     dInvT1_dInvRot1(2, 5) = est.second.at<double>(1, 0);
	//     dInvT1_dInvRot1(2, 8) = est.second.at<double>(2, 0);

	//     dRod = dRod.t();

	//     cv::Mat_<double> dDist_dRod = dDist_dInvT1 * dInvT1_dInvRot1 * dRod;
	//     jacobean.colRange(0, 3) += dDist_dRod * wTrans;


	//     // return gradient of rotational error
	//     cv::Mat_<double> dRotDiff = cv::Mat_<double>::zeros(9, 9);
	//     invRot2.row(0).copyTo(dRotDiff.row(0).colRange(0, 3));
	//     invRot2.row(1).copyTo(dRotDiff.row(1).colRange(0, 3));
	//     invRot2.row(2).copyTo(dRotDiff.row(2).colRange(0, 3));

	//     invRot2.row(0).copyTo(dRotDiff.row(3).colRange(3, 6));
	//     invRot2.row(1).copyTo(dRotDiff.row(4).colRange(3, 6));
	//     invRot2.row(2).copyTo(dRotDiff.row(5).colRange(3, 6));

	//     invRot2.row(0).copyTo(dRotDiff.row(6).colRange(6, 9));
	//     invRot2.row(1).copyTo(dRotDiff.row(7).colRange(6, 9));
	//     invRot2.row(2).copyTo(dRotDiff.row(8).colRange(6, 9));

	//     dRotDiff = dRotDiff.t();

	//     cv::Mat_<double> dTrace = cv::Mat_<double>::zeros(1, 9);
	//     dTrace(0, 0) = 1;
	//     dTrace(0, 4) = 1;
	//     dTrace(0, 8) = 1;

	//     cv::Mat_<double> dAngle = (180 / CV_PI * -1 / sqrt(3 - trace * trace + 2 * trace)) * dTrace * dRotDiff * dRod;

	//     jacobean.colRange(0, 3) += dAngle * wRot;
		
	// 	if(cutLoss)
	// 		jacobean *= 0.5 / loss;


	//     if(cv::sum(cv::Mat(jacobean != jacobean))[0] > 0) //check for NaNs
	//         return cv::Mat_<double>::zeros(1, 6);

	//     return jacobean;
	// }

	/**
	 * @brief Calculate the derivative of the loss w.r.t. the estimated pose.
	 * @param est Estimated pose (6 DoF).
	 * @param gt Ground truth pose (6 DoF).
	 * @param wRot Weight of rotation error.
	 * @param wTrans Weight of translation error.
	 * @param cut Apply soft clamping after this value.	 
	 * @return 1x6 Jacobean.
	 */
	cv::Mat_<double> dLoss(
		const dsacstar::pose_t& est, 
		const dsacstar::pose_t& gt,
		double xmin , 
		double xmax ,
		double step_size = 0.001, 
		double wRot = 1.0,
		double wTrans = 1.0,
		double cut = 100)
	{
		
/*

		cv::Mat_<double> est_r  = est.first.clone() ;
		cv::Mat_<double> gt_r   =  gt.first.clone() ;

*/
	    cv::Mat rot1, rot2, dRod;
	    cv::Rodrigues(est.first, rot1, dRod);
	    cv::Rodrigues(gt.first , rot2);
        // cout<<"est "<<est.first<<endl;
        // cout<<"gt "<<gt.first<<endl;
		// reconstruct the trasformations

		cv::Mat_<double> est_trans = cv::Mat_<double>::zeros(4,4) ; 
		
        est_trans = dsacstar::pose2trans(est);
        // cout<<"est_trans"<<est_trans<<endl;
        // cout<<"rot1 "<<rot1<<endl;
        // est_trans.colRange(0,3).rowRange(0,3) = rot1 ; 
        // cout<<"est trans "<<est_trans<<endl;
        // est_trans.col(3).rowRange(0,3) = est.second.clone();
		// est_trans(3,3) = 1 ;  


		cv::Mat_<double> gt_trans = cv::Mat_<double>::zeros(4,4) ; 
		gt_trans = dsacstar::pose2trans(gt);
        
        // gt_trans.colRange(0,3).rowRange(0,3) = rot2 ; 
        // gt_trans.col(3).rowRange(0,3) = gt.second.clone();

		// gt_trans(3,3) = 1 ; 
		// // cv::Mat_<double> chat_R_c = rot1.t() * rot2 ; 
		
		double homo_loss = dsacstar::loss(est_trans , gt_trans , xmin, xmax) ; 


		cv::Mat_<double> w_t_chat = est.second.clone();

	    cv::Mat_<double> w_t_c = gt.second.clone();
	    
		cv::Mat_<double> chat_t_c = (rot1.t()) * (w_t_c - w_t_chat) ; 

		cv::Mat_<double> jacobian = cv::Mat_<double>::zeros(1, 6);

		
	    if(homo_loss > MAXLOSS)
            {
            // cout<<"first jac"<<endl;
            return jacobian;
            }
		
		// estimate the gradient in rotation 
		cv::Mat_<double> est_rot1, est_rot2, est_rot3;
		// cv::Mat_<double> est_rv1 = est.first.clone()  ; 
		// cout<<"est before "<< est_rv1<<endl;
        // est_rv1(0) +=  step_size ;
        // cout<<"rot1 "<< rot1<<endl;
        
        
        
        
        dsacstar::pose_t est_rv1;
        est_rv1.first = est.first.clone();
        est_rv1.second = est.second.clone();

        est_rv1.first.row(0)+=step_size;
        cv::Mat_<double> est_dh1 = dsacstar::pose2trans(est_rv1);
		// cout<<"dh1 "<<est_dh1<<endl;
		
        
        double l_dr1 = dsacstar::loss(est_dh1 , gt_trans , xmin , xmax); 
		double dl_dr1 = (l_dr1 - homo_loss) / step_size ; 
        
		dsacstar::pose_t est_rv2;
        est_rv2.first = est.first.clone();
        est_rv2.second = est.second.clone();

        est_rv2.first.row(1)+=step_size;
        cv::Mat_<double> est_dh2 = dsacstar::pose2trans(est_rv2);
		// cout<<"dh1 "<<est_dh1<<endl;
		
		
		double l_dr2 = dsacstar::loss(est_dh2 , gt_trans , xmin , xmax); 
		double dl_dr2 = (l_dr2 - homo_loss) / step_size ; 

        
        dsacstar::pose_t est_rv3;
        est_rv3.first = est.first.clone();
        est_rv3.second = est.second.clone();

        est_rv3.first.row(2)+=step_size;
        cv::Mat_<double> est_dh3 = dsacstar::pose2trans(est_rv3);
		  
		double l_dr3 = dsacstar::loss(est_dh3 , gt_trans , xmin , xmax); 
		double dl_dr3 = (l_dr3 - homo_loss) / step_size ; 
        // cout<<"dl_dr2"<<dl_dr2<<"dl_dr1"<<dl_dr1<<"dl_dr3"<<dl_dr3<<endl;

		jacobian.col(0) =  dl_dr1 ; 
		jacobian.col(1) =  dl_dr2 ;
		jacobian.col(2) =  dl_dr3 ;
        // cout<<"jac = "<<jacobian<<endl;
		// estimate the gradient in translation
		cv::Mat_<double> est_dht1, est_dht2, est_dht3;

		dsacstar::pose_t est_t1;
        est_t1.first = est.first.clone();
        est_t1.second = est.second.clone();

        est_t1.second.row(0)+=step_size;
        est_dht1 = dsacstar::pose2trans(est_t1);
        // cout<<"est dht1"<<est_dht1<<endl;
		// cout<<"est "<<est.second<<endl;
        double l_dt1 = dsacstar::loss(est_dht1 , gt_trans , xmin , xmax); 
		double dl_dt1 = (l_dt1 - homo_loss) / step_size ; 
		


        dsacstar::pose_t est_t2;
        est_t2.first = est.first.clone();
        est_t2.second = est.second.clone();

        est_t2.second.row(1)+=step_size;
        est_dht2 = dsacstar::pose2trans(est_t2);
        
        double l_dt2 = dsacstar::loss(est_dht2 , gt_trans , xmin , xmax); 
		double dl_dt2 = (l_dt2 - homo_loss) / step_size ; 

		dsacstar::pose_t est_t3;
        est_t3.first = est.first.clone();
        est_t3.second = est.second.clone();

        est_t3.second.row(2)+=step_size;
        est_dht3 = dsacstar::pose2trans(est_t3);
        
        double l_dt3 = dsacstar::loss(est_dht3 , gt_trans , xmin , xmax); 
		double dl_dt3 = (l_dt3 - homo_loss) / step_size ; 
		  
		jacobian.col(3) =  dl_dt1 ; 
		jacobian.col(4) =  dl_dt2 ;
		jacobian.col(5) =  dl_dt3 ;

        // cout<<"jac 2"<<jacobian<<endl;
	    

	    // zero error, abort
	    // double tErr = cv::norm(invT1 - invT2);

	    // cv::Mat_<double> jacobean = cv::Mat_<double>::zeros(1, 6);
	    
	    // clamped loss, return zero gradient if loss is bigger than threshold
	    // double loss = wRot * rotErr + wTrans * tErr;
	    // bool cutLoss = false;

		// if(cutLoss)
		// 	jacobean *= 0.5 / loss;


	    if(cv::sum(cv::Mat(jacobian != jacobian))[0] > 0) //check for NaNs
            {
                // cout<<"1 jac"<<endl;
                return cv::Mat_<double >::zeros(1, 6);
            }
        // cout<<"last jac"<<endl;
	    return jacobian;
	}


}
