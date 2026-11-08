/** Teaching balances only. No EOS, plant coefficients or simulator backend. */
export function heatingRate(inputKw,outputKw,capacityKjPerK=20){
 if(![inputKw,outputKw,capacityKjPerK].every(Number.isFinite)||inputKw<0||outputKw<0||capacityKjPerK<=0) throw new RangeError('Finite nonnegative heat rates and positive capacity required');
 return (inputKw-outputKw)/capacityKjPerK;
}
export function phaseFractions(quality,liquidDensity=900,vapourDensity=5){
 if(![quality,liquidDensity,vapourDensity].every(Number.isFinite)||quality<0||quality>1||liquidDensity<=0||vapourDensity<=0) throw new RangeError('Admissible quality and positive densities required');
 const volume=(1-quality)/liquidDensity+quality/vapourDensity;
 return {density:1/volume,voidFraction:(quality/vapourDensity)/volume};
}
