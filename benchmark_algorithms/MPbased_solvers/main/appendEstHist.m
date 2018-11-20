% appends an estHist structure, for use with warm-starting

function outHist = appendEstHist(inHist,newHist)

  if (isempty(inHist))||(length(inHist.it)==0)
    outHist = newHist; 
  else
    outHist.xhat = [inHist.xhat,newHist.xhat];
    outHist.xvar = [inHist.xvar,newHist.xvar];
    outHist.Axhat = [inHist.Axhat,newHist.Axhat];
    outHist.phat = [inHist.phat,newHist.phat];
    outHist.pvar = [inHist.pvar,newHist.pvar];
    outHist.shat = [inHist.shat,newHist.shat];
    outHist.svar = [inHist.svar,newHist.svar];
    outHist.zhat = [inHist.zhat,newHist.zhat];
    outHist.zvar = [inHist.zvar,newHist.zvar];
    outHist.rhat = [inHist.rhat,newHist.rhat];
    outHist.rvar = [inHist.rvar,newHist.rvar];
    outHist.val =  [inHist.val;newHist.val];
    outHist.step = [inHist.step;newHist.step];
    outHist.stepMax = [inHist.stepMax;newHist.stepMax];
    outHist.pass = [inHist.pass;newHist.pass];
    outHist.scaleFac = [inHist.scaleFac;newHist.scaleFac];
    outHist.it = [inHist.it;inHist.it(end)+newHist.it];
  end
