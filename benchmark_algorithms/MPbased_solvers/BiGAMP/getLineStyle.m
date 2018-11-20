function lineStyle = getLineStyle(algName)
%Simple approach to achieve consistent line styles for algorithms. Returns
%the line style to use for algName

switch algName
    
    %New ones for P-BiG-AMP
    case 'P-BiG-AMP',
        lineStyle = 'bd-';
        
    case 'EM-P-BiG-AMP',
        lineStyle = 'm+-';
        
    case 'WSS-TLS',
        lineStyle = 'rx-';
        
    case 'Oracles',
        lineStyle = 'ko-';
        
    %BiG-AMP
    case 'BiG-AMP',
        lineStyle = 'bd-';
        
    case 'BiG-AMP-1',
        lineStyle = 'bd-';
        
    case 'BiG-AMP-2',
        lineStyle = 'bo-';
        
    case 'BiG-AMP X2',
        lineStyle = 'bo-';
        
    case 'BiG-AMP (Augmented Matrix)',
        lineStyle = 'bo-';
        
    case 'EM-BiG-AMP',
        lineStyle = 'bh-';
        
    case 'EM-BiG-AMP-1',
        lineStyle = 'bh-';
        
    case 'EM-BiG-AMP-2',
        lineStyle = 'b-*';
        
    case 'EM-BiG-AMP-2 (Rank Contraction)',
        lineStyle = 'b-v';
        
    case 'EM-BiG-AMP (AICc)',
        lineStyle = 'b-^';
        
            case 'EM-BiG-AMP (pen. log-like)',
        lineStyle = 'b-^';
        
    case 'EM-BiG-AMP (Rank Contraction)',
        lineStyle = 'b-v';
        
    case 'EM-BiG-AMP Lite',
        lineStyle = 'b-*';
        
    case 'BiG-AMP Lite',
        lineStyle = 'bs-';
        
    case 'BiG-AMP SVD',
        lineStyle = 'b*-';
        
    case 'BiG-AMP Lite SVD',
        lineStyle = 'k*-';
        
    case 'BiG-AMP Lite v4',
        lineStyle = 'rv-';
        
    case 'BiG-AMP Lite v5',
        lineStyle = 'gv-';
        
    case 'BiG-AMP Fast',
        lineStyle = 'bs-';
        
    case 'SVT',
        lineStyle = 'b.-';
        
    case 'GROUSE',
        lineStyle = 'rx-';
        
    case 'GRASTA',
        lineStyle = 'rx-';
        
    case 'Matrix ALPS',
        lineStyle = 'ko-';
        
    case 'Inexact ALM',
        lineStyle = 'g^-';
        
    case 'IALM',
        lineStyle = 'g^-';
        
    case 'IALM-1',
        lineStyle = 'g^-';
        
    case 'IALM-2',
        lineStyle = 'gv-';
        
    case 'VSBL',
        lineStyle = 'cs-';
        
    case 'LMaFit',
        lineStyle = 'mp-';
        
    case 'LMaFit SVD',
        lineStyle = 'm+-';
        
    case 'OptSpace',
        lineStyle = 'ro-';
        
    case 'SPAMS',
        lineStyle = 'g^-';
        
    case 'ER-SpUD (SC)',
        lineStyle = 'mp-';
        
    case 'ER-SpUD (proj)',
        lineStyle = 'mp-';
        
    case 'K-SVD',
        lineStyle = 'rx-';
        
    otherwise,
        lineStyle = 'k--';
        warning('Algorithm unknown') %#ok<WNTAG>
end