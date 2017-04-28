//
//  kMeans.swift
//  clusters
//
//  Created by istros2015 on 28/04/2017.
//  Copyright © 2017 ___Istros Anlagen sro___. All rights reserved.
//

import Foundation

protocol ModNN {
    func sigmoid(x: Float)->(Float)
    func dsigmoid(x: Float)->(Float)
    func logistic(x: Float)->(Float)
    func dlogistic(x: Float)->(Float)
    func truncsigm(x:Float, coef:Float)->Float
    func logcap (cap: Float)->(Float)
    func zscore(x : Float, e : Float, s : Float)->(Float)
    func statMono(ar : Array<Float>)->([String : Float])
    func calculSR (corr:Float, pop:UInt32)->(Float)
}

extension ModNN {
    func sigmoid(x: Float)->(Float){
        return tanh(x)
    }
    // derivative of our sigmoid function
    func dsigmoid(x: Float)->(Float){
        //  return 1.0 - x**2.0
        return 1.0 - pow(x,2.0)
    }
    func logistic(x: Float)->(Float){
        //    return tanh(x)
        return 1.0 / (1.0 + exp(-x))
    }
    // derivative of our sigmoid function
    func dlogistic(x: Float)->(Float){
        //  return 1.0 - x**2.0
        //  return 1.0 - pow(x,2.0)
        return x * (1.0 - x)
    }
    func truncsigm(x:Float, coef:Float)->Float {
        if coef > 0.0 {
            return coef * tanh(x/coef)
        } else {
            return 0.0
        }
    }
    func logcap (cap: Float)->(Float) {
        return log10(10.0 + (cap / 5.0))
    }
    func zscore(x : Float, e : Float, s : Float)->(Float) {
        if s > 0.0 {
            return (x - e) / s
        }
        return 0.0
    }
    func statMono(ar : Array<Float>)->([String : Float]) {
        var moy : Float = 0.0
        var sum : Float = 0.0
        var median : Float = 0.0
        var max : Float = 0.0
        var min : Float = 0.0
        var sx2 : Float = 0.0
        //     var var2 : Float = 0.0
        let n : Float = Float(ar.count)
        for ivar in ar {
            sum += ivar
            sx2 += (ivar * ivar)
            if ivar == ar[0] {
                max = ivar
                min = ivar
            }
            if ivar < min {
                min = ivar
            }
            max = (max < ivar) ? ivar : max
        }
        moy = sum / n
        /*
         for ivar in ar {
         var2 += (ivar - moy)*(ivar - moy)/n
         }
         */
        let vax = (sx2 / n) - (moy * moy)
        let sigx = sqrt(vax)
        let sar = ar.sort { $0 < $1 }
        let k = sar.count
        if k % 2 == 0 {
            median = (sar[k/2 - 1] + sar[k/2]) / 2.0
        } else {
            median = sar[k/2]
        }
        let dico = [ "moy" : moy,
                     "median" : median,
                     "min" : min,
                     "max" : max,
                     "ec" : sigx]
        return dico
    }
    
    func calculSR (corr:Float, pop:UInt32)->(Float) {
        var sr : Float = 0.0
        if pop > 3 {
            if pop > 100 {
                sr = 1.96 / sqrtf(Float(pop-1))
            }
            else {
                sr = tanhf(1.96/sqrtf(Float(pop-3)))
            }
        }
        return sr
    }
    
    func transmeta(vari : String, imeta : Float)->(Float) {
        let itrans : Float
        switch vari {
        case "etafiNY.roic":
            itrans = truncsigm(imeta,coef: 50.0)
        case "etafiNY.roeci":
            itrans = truncsigm(imeta,coef: 50.0)
        case "etafiMoy.fcfy":
            itrans = truncsigm(imeta,coef: 40.0)
        case "etafiMoy.potg":
            itrans = truncsigm(imeta,coef: 30.0)
        case "etafiVar.eps":
            itrans = truncsigm(imeta,coef: 40.0)
        case "etafiVar.roce":
            itrans = truncsigm(imeta,coef: 40.0)
        case "etafiVar.eva":
            itrans = truncsigm(imeta,coef: 40.0)
        case "etafiVar.roeci":
            itrans = truncsigm(imeta,coef: 40.0)
        case "etafiVar.revenue":
            itrans = truncsigm(imeta,coef: 40.0)
        case "cap":
            itrans = logcap(imeta)
        default:
            itrans = imeta
        }
        return itrans
    }
}

class kMeans : ModNN {
    var nom : String = "kMeans"
    var nv : Int = 7
    var ca : Float = 0.0
    var cb : Float = 1.0
    var add : String
    var ibase = NSArray()
    var pat = Array<Array<Float>>()
    var clusters = Array<Array<Array<Float>>>()
    var indclu = Array<Array<Int>>()
    var barycentres = Array<Array<Float>>()
    var k : Int = 5
    
    var addR2data : String?
    
    let metric1 : Array<String> = ["etafiNY.pbr", "etafiNY.potg", "etafiVar.eva", "levScore2", "etafiVar.eps", "etafiVar.roce"]
    let metric2 : Array<String> = ["etafiNY.roeci", "etafiNY.roic", "etafiMoy.fcfy", "etafiNY.rend", "etafiNY.evnopat", "n1Score", "trc4", "levScore2", "etafiVar.roce", "etafiVar.eps", "etafiVar.eva", "zlatoScore",]
    var metric = Array<String>();
    
    var minInputs = Array<Float>()
    var maxInputs = Array<Float>()
    var maxminInputs = Array<Float>()
    
    init (k : Int) {
        self.k = k
        add = "~/Finance/Aetius/exportPopLarge.plist"
        self.addR2data = "~/Finance/Aetius/testR2A" + nom + ".csv"
        ca = 0.0
        cb = 1.0
        metric = metric2
        ibase = base(add)
        setMinMax(ibase)
        pat = chargeData(add)
    }
    
    func initClusters(k : Int) {
        
        let ari0 = Array(count:1, repeatedValue:Int(0))
        let ar0 = Array(count:nv, repeatedValue:Float(0.0))
        let ar00 = [ar0]
        clusters = Array(count:k, repeatedValue:ar00)
        barycentres = Array(count:k, repeatedValue:ar0)
        indclu = Array(count:k, repeatedValue:ari0)
        var lastip : Array<Array<Float>> = [pat[0]]
        
        for i in 0...k-1 {
            var test = false
            var ip = pat[0]
            var irang = 0
            if i>0 {
                while test == false {
                    irang = aleaRang(pat.count-1)
                    ip = pat[irang]
                    let dli = distglobale(ip, aa: lastip)
                    let dig = distglobale(ip, aa: pat)
                    let rdip = dli / dig
                    test = (rdip > 0.6)
                }
                if test {
                    clusters[i].append(ip)
                    clusters[i].removeFirst()
                    barycentres[i] = ip
                    indclu[i].append(irang)
                    indclu[i].removeFirst()
                }
            }
            else {
                clusters[i].append(ip)
                clusters[i].removeFirst()
                barycentres[i] = ip
                indclu[i].append(irang)
                indclu[i].removeFirst()
            }
            lastip.append(ip)
        }
    }
    
    func affecte(a: Array<Float>, indice : Int, inout barys : Array<Array<Float>>) {
        
        var (indm, dmin) : (Int, Float) = (0, 999999.0)
        
        for (index, item) in barys.enumerate() {
            let d = dist(a, b: item)
            (indm, dmin) = (d < dmin) ? (index, d) : (indm, dmin)
        }
        /*    for item in indclu[indm] {
         item != indice
         }
         */
        let test = indclu[indm].reduce(true) {
            initial, next in
            return initial && (next != indice)
        }
        if test {
            clusters[indm].append(a)
            indclu[indm].append(indice)
        }
        
        barycentres[indm] = barycentre(clusters[indm])
    }
    
    func run() {
        var test2 = false
        while (!test2) {
            initClusters(k)
            for (index, item) in pat.enumerate() {
                affecte(item, indice: index, barys: &barycentres)
            }
            var cpt = 0
            for j in barycentres.indices {
                let ni = clusters[j].count
                if ni >= 3 {cpt += 1}
            }
            if cpt == k {test2 = true}
        }
        detailclusters(barycentres)
    }
    
    func base (add : String)->(NSArray) {
        let nadd = (add as NSString).stringByExpandingTildeInPath
        let dat = NSArray(contentsOfFile:nadd)
        return dat!
    }
    
    func setMinMax(base : NSArray) {
        var pat = Array<Array<Float>>()
        for item in base {
            let item2  = item as! NSDictionary
            var iar = Array<Float>()
            for vari : String in metric {
                let fvari = item2.objectForKey(vari) as! Float
                let fvi = transmeta(vari, imeta: fvari)
                iar.append(fvi)
            }
            let eva = item2.objectForKey("etafiNY.eva") as! Float
            let ev = item2.objectForKey("ev") as! Float
            var eer : Float = 0.0;
            if ev > 0.0 {
                eer = eva/ev
            }
            iar.append(eer)
            nv = iar.count
            pat.append(iar)
        }
        
        for p in 0...pat.count-1{
            var inputs = pat[p]
            if p == 0 {
                maxInputs = pat[p]
                minInputs = pat[p]
            }
            for j in 0...inputs.count-1 {
                if inputs[j] > maxInputs[j] {
                    maxInputs[j] = inputs[j]
                } else
                    if inputs[j] < minInputs[j] {
                        minInputs[j] = inputs[j]
                }
            }
        }
        let ni = pat[0].count
        maxminInputs = [Float](count: ni, repeatedValue:0.0)
        for j in 0...ni-1 {
            maxminInputs[j] = maxInputs[j] - minInputs[j]
        }
    }
    
    func filteredBase (secteur : String)->(NSArray) {
        let pred : NSPredicate = NSPredicate.init(format:"secteur BEGINSWITH[cd] %@", argumentArray:[secteur])
        let fb = ibase.filteredArrayUsingPredicate(pred)
        return fb
    }
    
    func cafilteredBase (casecteurs : [String])->(NSArray) {
        var arpred : Array<NSPredicate> = Array()
        for index in casecteurs.indices {
            let predi : NSPredicate = NSPredicate.init(format:"secteur BEGINSWITH[cd] %@", argumentArray:[casecteurs[index]])
            arpred.append(predi)
        }
        let superPred = NSCompoundPredicate.init(orPredicateWithSubpredicates : arpred)
        let fb = ibase.filteredArrayUsingPredicate(superPred)
        return fb
    }
    
    func barycentre (base : NSArray)->(Array<Float>) {
        let pat : Array<Array<Float>> = chargeData2(base)
        let n = pat.count
        var psi = [Float](count: nv, repeatedValue:0.0)
        for i in 0..<nv {
            var ssi : Float = 0.0
            for fi : Array<Float> in pat {
                ssi += fi[i]
                psi[i] = ssi / Float(n)
            }
        }
        return psi
    }
    
    func barycentre (pat : Array<Array<Float>>)->(Array<Float>) {
        //    let pat : Array<Array<Float>> = chargeData2(base)
        let n = pat.count
        var psi = [Float](count: nv, repeatedValue:0.0)
        for i in 0..<nv {
            var ssi : Float = 0.0
            for fi : Array<Float> in pat {
                ssi += fi[i]
                psi[i] = ssi / Float(n)
            }
        }
        return psi
    }
    
    func barycentre (secteur : String)->([Float]) {
        let cb = filteredBase(secteur)
        return barycentre(cb)
    }
    
    func barycentre (casecteurs : [String])->([Float]) {
        let cb = cafilteredBase(casecteurs)
        return barycentre(cb)
    }
    
    func chargeData(add : String)->(Array<Array<Float>>) {
        let nadd = (add as NSString).stringByExpandingTildeInPath
        let dat = NSArray(contentsOfFile:nadd)
        var pat = Array<Array<Float>>()
        
        for item in dat! {
            let item2  = item as! NSDictionary
            var iar = Array<Float>()
            for vari : String in metric {
                let fvari = item2.objectForKey(vari) as! Float
                let fvi = transmeta(vari, imeta: fvari)
                iar.append(fvi)
            }
            let eva = item2.objectForKey("etafiNY.eva") as! Float
            let ev = item2.objectForKey("ev") as! Float
            var eer : Float = 0.0;
            if ev > 0.0 {
                eer = eva/ev
            }
            iar.append(eer)
            pat.append(iar)
        }
        let pat2 = normalise(pat)
        return pat2
    }
    
    func chargeData2(base : NSArray)->(Array<Array<Float>>) {
        var pat = Array<Array<Float>>()
        for item in base {
            let item2  = item as! NSDictionary
            var iar = Array<Float>()
            for vari : String in metric {
                let fvari = item2.objectForKey(vari) as! Float
                let fvi = transmeta(vari, imeta: fvari)
                iar.append(fvi)
            }
            let eva = item2.objectForKey("etafiNY.eva") as! Float
            let ev = item2.objectForKey("ev") as! Float
            var eer : Float = 0.0;
            if ev > 0.0 {
                eer = eva/ev
            }
            iar.append(eer)
            pat.append(iar)
        }
        let pat2 = normalise(pat)
        return pat2
    }
    
    func normalise(patterns:Array<Array<Float>>)->(Array<Array<Float>>) {
        
        var npatterns = Array<Array<Float>>()
        
        for p in 0...patterns.count-1{
            var inputs = patterns[p]
            
            for j in 0...inputs.count-1 {
                inputs[j] =  (inputs[j] - minInputs[j]) / maxminInputs[j]
            }
            npatterns.append(inputs)
        }
        
        return npatterns
    }
    
    func pop (secteur : String)->(Int) {
        let cb = filteredBase(secteur)
        let p = cb.count
        return p
    }
    
    func pop (casecteurs : [String])->(Int) {
        let cb = cafilteredBase(casecteurs)
        let p = cb.count
        return p
    }
    
    func dist(a: Array<Float>, b: Array<Float>)->(Float) {
        var ssi : Float = 0.0
        for (index, number) in a.enumerate() {
            let fbi = b[index]
            let si = (number - fbi) * (number - fbi)
            ssi += si
        }
        return sqrtf(ssi)
    }
    
    func distglobale(a: Array<Float>, aa : Array<Array<Float>>)->(Float) {
        let ac = aa.count
        if ac <= 1 {
            return 0.0
        } else {
            var ssi : Float = 0.0
            for item in aa {
                let d = dist(a, b: item)
                ssi += d
            }
            return ssi/Float(ac) // ?
        }
    }
    
    func di (secteur : String)->(Float) {
        //    let baryi = barycentre(secteur)
        let cb = filteredBase(secteur)
        let pati = chargeData2(cb)
        let baryi = barycentre(pati)
        let di = distglobale(baryi, aa: pati)
        return di
    }
    
    func di (casecteurs : [String])->(Float) {
        let baryi = barycentre(casecteurs)
        let cb = cafilteredBase(casecteurs)
        let pati = chargeData2(cb)
        let di = distglobale(baryi, aa: pati)
        return di
    }
    
    func di (barys : Array<Array<Float>>, indice : Int)->(Float) {
        let pati = clusters[indice]
        let di = distglobale(barys[indice], aa: pati)
        return di
    }
    
    func rdi (secteur : String)->(Float) { // ratio entre la dist. moyenne du cluster / dist. moy de tous les points au barycentre du clister
        let baryi = barycentre(secteur)
        let cb = filteredBase(secteur)
        let pati = chargeData2(cb)
        let di = distglobale(baryi, aa: pati)
        let dig = distglobale(baryi, aa: pat)
        let rdi = di / dig
        return rdi
    }
    
    func rdi (casecteurs : [String])->(Float) {
        let baryi = barycentre(casecteurs)
        let cb = cafilteredBase(casecteurs)
        let pati = chargeData2(cb)
        let di = distglobale(baryi, aa: pati)
        let dig = distglobale(baryi, aa: pat)
        let rdi = di / dig
        return rdi
    }
    
    func rdi (barys : Array<Array<Float>>, indice : Int)->(Float) {
        let pati = clusters[indice]
        let bary = barycentres[indice]
        let di = distglobale(bary, aa: pati)
        let dig = distglobale(bary, aa: pat)
        let rdi = di / dig
        return rdi
    }
    
    func maxDB (barys : Array<Array<Float>>, indice: Int)->(Float) {
        var max : Float = 0.0
        let ibary = barys[indice]
        let dii = di(barys, indice : indice)
        for (index, jbary) in barys.enumerate() {
            if index != indice {
                let dij = di(barys, indice: index)
                let dbaryij = dist(ibary, b: jbary)
                if dbaryij != 0.0 {
                    let ratij = ( dii + dij) / dbaryij
                    if ratij > max {
                        max = ratij
                    }
                }
            }
        }
        return max
    }
    
    func DBindex (barys : Array<Array<Float>>)->(Float) { // Davies–Bouldin index, the smallest the best !
        let n = k
        assert (k>0,"k doit >0 !")
        var sum : Float = 0.0
        for (index, _) in barys.enumerate() {
            sum += maxDB(barys, indice: index)
        }
        return sum / Float(n)
    }
    
    func aleaRang(a: Int, b:Int) -> (Int) {
        let k : Int32 = rand()
        let max = RAND_MAX
        let f1 : Float = Float(k) / Float(max)
        let output = Float(b-a) * f1 + Float(a)
        return Int(output)
    }
    
    func aleaRang(b:Int) -> (Int) {
        let bb = UInt32(b)
        let rang2 = arc4random_uniform(bb)
        return Int(rang2)
    }
    
    func distToBarys(a: Array<Float>, barys : Array<Array<Float>>)->(Array<(Int, Float)>) {
        var ard = Array<(Int, Float)>()
        var (indm, dmin) : (Int, Float) = (0, 0.0)
        for (index, item) in barys.enumerate() {
            let d = dist(a, b: item)
            ard.append((index, d))
            (indm, dmin) = (d < dmin) ? (index, d) : (indm, dmin)
        }
        return ard
    }
    
    func ldpToBarys(a: Array<Float>, barys : Array<Array<Float>>)->(Array<(Int, Float)>) {
        let dists = distToBarys(a, barys: barys)
        let n = dists.count
        assert(n>1,"n doit >= 1 !")
        let ad = dists.map {$0.1}
        let sum = ad.reduce(0.0, combine: +)
        //   sum /= Float(n)
        var arco = Array<(Int, Float)>()
        for (index, _) in barys.enumerate() {
            let d = dists[index].1 // dist(a, b: item)
            let coefi = (1.0 - (d / sum)) /  Float(n-1)
            arco.append((index, coefi))
        }
        let ldpsum = arco.map{$0.1}.reduce(0.0, combine: +)
        assert(fabs(ldpsum  - 1.0) < 0.0001, " ldpsum doit == 1.0 !")
        return arco
    }
    
    func describe(clus : Array<Array<Int>>, indice: Int) {
        let indclui = clus[indice]
        let n = indclui.count
        let rdii = rdi(barycentres, indice : indice)
        print("barycentre \(indice)  -> pop: \(n), rdi: \(rdii)")
        assert(n>1, "n doit >1 !")
        for vari in metric {
            var ari = Array<Float>()
            for indexi in indclui.indices {
                let indi = indclui[indexi]
                let item2 = ibase[indi] as! NSDictionary
                let fvari = item2.objectForKey(vari) as! Float
                ari.append(fvari)
            }
            let idic = statMono(ari)
            print("\(vari) -> moy :\(idic["moy"]!)")
        }
    }
    
    func detailclusters(barycentres : Array<Array<Float>>) {
        print("DETAILS CLUSTERISATION")
        print("***")
        print("\(metric)" + "+ eer")
        print("dims : \(nv)")
        print("nb de clusters : \(k)")
        let lb = barycentres
        for i in barycentres.indices {
            let rdii = rdi(barycentres, indice : i)
            //  print("barycentre \(i) : \(ba)\n   -> pop: \(clusters[i].count), rdi: \(rdii)")
            print("barycentre \(i)  -> pop: \(clusters[i].count), rdi: \(rdii)")
        }
        let dbi = DBindex(barycentres)
        print("\nDavies-Bouldin index : \(dbi)")
        for index in lb.indices {
            let baryi = lb[index]
            let ldi = lb.map { dist(baryi, b: $0)}
            print("bary\(index) : \(ldi)")
        }
        print("***********")
        for index in lb.indices {
            let indclui = indclu[index]
            print("bary\(index) :")
            for indexi in indclui.indices {
                let indi = indclui[indexi]
                let oi1 = ibase[indi] as! NSDictionary
                let inom = oi1.valueForKey("nom")!
                print("\(indi) : \(inom)")
            }
            
            //      let ldi = lb.map { base.objectAtIndex:$0.valueForKey:@"nom"}
            //      let dic = ibase.objectAtIndex:index
        }
        for index in lb.indices {
            //   let indclui = indclu[index]
            print("***\n")
            describe(indclu, indice: index)
        }
        print("***********")
    }

}
