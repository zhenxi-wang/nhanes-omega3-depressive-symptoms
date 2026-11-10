args <- commandArgs(trailingOnly=TRUE)
v <- args[1]
if(length(args)>1) .libPaths(c(args[2],.libPaths()))
suppressPackageStartupMessages(library(survey))
options(survey.lonely.psu='fail',warn=1)
spec <- read.csv(file.path(v,'models.csv'),stringsAsFactors=FALSE)
pyb <- read.csv(file.path(v,'python_coefficients.csv'),stringsAsFactors=FALSE,check.names=FALSE)
pyv <- read.csv(file.path(v,'python_covariances.csv'),stringsAsFactors=FALSE)
results <- list(); coefficients <- list()
for(i in seq_len(nrow(spec))){
 sp <- spec[i,];d <- read.csv(file.path(v,'inputs',sp$path),check.names=FALSE)
 d$weight[is.na(d$weight)] <- 0
 xs <- grep('^x[0-9]+$',names(d),value=TRUE)
 des <- svydesign(ids=~psu,strata=~strata,weights=~weight,data=d,nest=TRUE)
 sub <- subset(des,domain==1)
 fm <- as.formula(paste('outcome ~ -1 +',paste(xs,collapse='+')))
 fit <- svyglm(fm,design=sub,family=if(sp$kind=='OR') quasibinomial() else gaussian(),control=glm.control(epsilon=1e-14,maxit=100))
 b <- coef(fit);vv <- vcov(fit)
 pb <- pyb[pyb$case==sp$case & pyb$mi==sp$mi,];pv <- pyv[pyv$case==sp$case & pyv$mi==sp$mi,]
 mb <- max(abs(unname(b[pb$term])-pb$beta));mv <- max(abs(vv[cbind(match(pv$term1,names(b)),match(pv$term2,names(b)))]-pv$covariance))
 mse <- max(abs(sqrt(diag(vv))[pb$term]-pb$SE))
 results[[i]] <- data.frame(case=sp$case,mi=sp$mi,n=nrow(model.frame(fit)),parameters=length(b),design_df=degf(des),max_abs_coefficient_difference=mb,max_abs_covariance_difference=mv,max_abs_SE_difference=mse)
 coefficients[[i]] <- data.frame(case=sp$case,mi=sp$mi,term=names(b),beta=unname(b),variance=diag(vv))
 if(i%%20==0)cat('R SURVEY VALIDATION',i,'/',nrow(spec),'\n')
}
rr <- do.call(rbind,results);cc <- do.call(rbind,coefficients)
write.csv(rr,file.path(v,'R_python_comparison.csv'),row.names=FALSE)
write.csv(cc,file.path(v,'R_coefficients.csv'),row.names=FALSE)
pool_one <- function(q,u){
 m<-length(q);qb<-mean(q);ub<-mean(u);bb<-var(q);tt<-ub+(1+1/m)*bb;la<-(1+1/m)*bb/tt
 old<-(m-1)/max(la,1e-15)^2;obs<-(16/18)*15*(1-la);dd<-1/(1/old+1/obs)
 data.frame(OR=exp(qb),lo=exp(qb-qt(.975,dd)*sqrt(tt)),hi=exp(qb+qt(.975,dd)*sqrt(tt)),P=2*pt(-abs(qb/sqrt(tt)),dd),U=ub,B=bb,T=tt,df=dd)
}
pooled <- lapply(c('main_dep','main_dep_sev','diet_HEI'),function(k){
 term<-unique(pyb$term[pyb$case==k & pyb$label=='O3I']);r<-cc[cc$case==k & cc$term==term,]
 cbind(case=k,M=nrow(r),pool_one(r$beta,r$variance))
})
write.csv(do.call(rbind,pooled),file.path(v,'R_pooled_results.csv'),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(v,'R_sessionInfo.txt'))
writeLines(capture.output(citation('survey',lib.loc=.libPaths()[1])),file.path(v,'survey_citation.txt'))
if(any(rr$design_df!=15))stop('Unexpected design degrees of freedom')
if(max(rr$max_abs_coefficient_difference)>1e-6||max(rr$max_abs_covariance_difference)>1e-6)stop('Validation tolerance exceeded')
cat('R SURVEY VALIDATION PASSED',nrow(rr),'models. Maximum differences:',max(rr$max_abs_coefficient_difference),max(rr$max_abs_covariance_difference),'\n')
