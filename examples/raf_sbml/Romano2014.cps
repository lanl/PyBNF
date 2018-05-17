<?xml version="1.0" encoding="UTF-8"?>
<!-- generated with COPASI 4.22 (Build 170) (http://www.copasi.org) at 2018-04-06 15:15:20 UTC -->
<?oxygen RNGSchema="http://www.copasi.org/static/schema/CopasiML.rng" type="xml"?>
<COPASI xmlns="http://www.copasi.org/static/schema" versionMajor="4" versionMinor="22" versionDevel="170" copasiSourcesModified="0">
  <ListOfFunctions>
    <Function key="Function_50" name="Function for v1" type="UserDefined" reversible="unspecified">
      <Expression>
        k1*RasGTP*pRaf_1i/(Km1+pRaf_1i)/default_compartment
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_333" name="Km1" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_334" name="RasGTP" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_335" name="default_compartment" order="2" role="volume"/>
        <ParameterDescription key="FunctionParameter_336" name="k1" order="3" role="constant"/>
        <ParameterDescription key="FunctionParameter_337" name="pRaf_1i" order="4" role="substrate"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_51" name="Function for v2" type="UserDefined" reversible="unspecified">
      <Expression>
        (k2a*LATS1a*Raf_1/(Km2a+Raf_1)+k2b*Kin*Raf_1/(Km2b+Raf_1))/default_compartment
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_346" name="Kin" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_347" name="Km2a" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_348" name="Km2b" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_349" name="LATS1a" order="3" role="modifier"/>
        <ParameterDescription key="FunctionParameter_350" name="Raf_1" order="4" role="substrate"/>
        <ParameterDescription key="FunctionParameter_351" name="default_compartment" order="5" role="volume"/>
        <ParameterDescription key="FunctionParameter_352" name="k2a" order="6" role="constant"/>
        <ParameterDescription key="FunctionParameter_353" name="k2b" order="7" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_52" name="Function for v3" type="UserDefined" reversible="unspecified">
      <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Function_52">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-06T08:53:15Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

      </MiriamAnnotation>
      <Expression>
        V3*Raf_1/(Km3+Raf_1)*((1+Fa*(ppERK/Ka))/(1+ppERK/Ka))/default_compartment
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_343" name="Fa" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_362" name="Ka" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_363" name="Km3" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_364" name="Raf_1" order="3" role="substrate"/>
        <ParameterDescription key="FunctionParameter_365" name="V3" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_366" name="default_compartment" order="5" role="volume"/>
        <ParameterDescription key="FunctionParameter_367" name="ppERK" order="6" role="modifier"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_53" name="Function for v4" type="UserDefined" reversible="unspecified">
      <Expression>
        V4*Raf_1a/(Km4+Raf_1a)/default_compartment
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_345" name="Km4" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_344" name="Raf_1a" order="1" role="substrate"/>
        <ParameterDescription key="FunctionParameter_332" name="V4" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_375" name="default_compartment" order="3" role="volume"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_54" name="Function for v5a" type="UserDefined" reversible="unspecified">
      <Expression>
        k5f*pMST2i*pRaf_1i/default_compartment
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_380" name="default_compartment" order="0" role="volume"/>
        <ParameterDescription key="FunctionParameter_381" name="k5f" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_382" name="pMST2i" order="2" role="substrate"/>
        <ParameterDescription key="FunctionParameter_383" name="pRaf_1i" order="3" role="substrate"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_55" name="Function for v5b" type="UserDefined" reversible="unspecified">
      <Expression>
        k5r*pMi_pRi/default_compartment
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_246" name="default_compartment" order="0" role="volume"/>
        <ParameterDescription key="FunctionParameter_388" name="k5r" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_389" name="pMi_pRi" order="2" role="substrate"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_56" name="Function for v6a" type="UserDefined" reversible="unspecified">
      <Expression>
        k6f*MST2*pRaf_1i/default_compartment
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_394" name="MST2" order="0" role="substrate"/>
        <ParameterDescription key="FunctionParameter_395" name="default_compartment" order="1" role="volume"/>
        <ParameterDescription key="FunctionParameter_396" name="k6f" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_397" name="pRaf_1i" order="3" role="substrate"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_57" name="Function for v6b" type="UserDefined" reversible="unspecified">
      <Expression>
        k6r*M_pRi/default_compartment
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_315" name="M_pRi" order="0" role="substrate"/>
        <ParameterDescription key="FunctionParameter_402" name="default_compartment" order="1" role="volume"/>
        <ParameterDescription key="FunctionParameter_403" name="k6r" order="2" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_58" name="Function for v7" type="UserDefined" reversible="unspecified">
      <Expression>
        k7*AKTa*MST2*(1+Kact*RasGTP)/(Km7+MST2)/default_compartment
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_411" name="AKTa" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_412" name="Kact" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_413" name="Km7" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_414" name="MST2" order="3" role="substrate"/>
        <ParameterDescription key="FunctionParameter_415" name="RasGTP" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_416" name="default_compartment" order="5" role="volume"/>
        <ParameterDescription key="FunctionParameter_417" name="k7" order="6" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_59" name="Function for v8" type="UserDefined" reversible="unspecified">
      <Expression>
        k8*PP2A*pMST2i/(Km8+pMST2i)/default_compartment
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_407" name="Km8" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_393" name="PP2A" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_425" name="default_compartment" order="2" role="volume"/>
        <ParameterDescription key="FunctionParameter_426" name="k8" order="3" role="constant"/>
        <ParameterDescription key="FunctionParameter_427" name="pMST2i" order="4" role="substrate"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_60" name="Function for v9" type="UserDefined" reversible="unspecified">
      <Expression>
        k9*MST2*MST2/default_compartment
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_410" name="MST2" order="0" role="substrate"/>
        <ParameterDescription key="FunctionParameter_264" name="default_compartment" order="1" role="volume"/>
        <ParameterDescription key="FunctionParameter_433" name="k9" order="2" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_61" name="Function for v10" type="UserDefined" reversible="unspecified">
      <Expression>
        k10*PP2A*MST2a/(Km10+MST2a)/default_compartment
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_439" name="Km10" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_440" name="MST2a" order="1" role="substrate"/>
        <ParameterDescription key="FunctionParameter_441" name="PP2A" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_442" name="default_compartment" order="3" role="volume"/>
        <ParameterDescription key="FunctionParameter_443" name="k10" order="4" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_62" name="Function for v11a" type="UserDefined" reversible="unspecified">
      <Expression>
        k11f*MST2a*F1A/default_compartment
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_437" name="F1A" order="0" role="substrate"/>
        <ParameterDescription key="FunctionParameter_449" name="MST2a" order="1" role="substrate"/>
        <ParameterDescription key="FunctionParameter_450" name="default_compartment" order="2" role="volume"/>
        <ParameterDescription key="FunctionParameter_451" name="k11f" order="3" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_63" name="Function for v11b" type="UserDefined" reversible="unspecified">
      <Expression>
        k11r*Ma_F1A/default_compartment
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_409" name="Ma_F1A" order="0" role="substrate"/>
        <ParameterDescription key="FunctionParameter_456" name="default_compartment" order="1" role="volume"/>
        <ParameterDescription key="FunctionParameter_457" name="k11r" order="2" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_64" name="Function for v12a" type="UserDefined" reversible="unspecified">
      <Expression>
        k12f*MST2*F1A/default_compartment
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_462" name="F1A" order="0" role="substrate"/>
        <ParameterDescription key="FunctionParameter_463" name="MST2" order="1" role="substrate"/>
        <ParameterDescription key="FunctionParameter_464" name="default_compartment" order="2" role="volume"/>
        <ParameterDescription key="FunctionParameter_465" name="k12f" order="3" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_65" name="Function for v12b" type="UserDefined" reversible="unspecified">
      <Expression>
        k12r*M_F1A/default_compartment
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_262" name="M_F1A" order="0" role="substrate"/>
        <ParameterDescription key="FunctionParameter_470" name="default_compartment" order="1" role="volume"/>
        <ParameterDescription key="FunctionParameter_471" name="k12r" order="2" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_66" name="Function for v13" type="UserDefined" reversible="unspecified">
      <Expression>
        V13*M_F1A/(Km13+M_F1A)/default_compartment
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_476" name="Km13" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_477" name="M_F1A" order="1" role="substrate"/>
        <ParameterDescription key="FunctionParameter_478" name="V13" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_479" name="default_compartment" order="3" role="volume"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_67" name="Function for v14" type="UserDefined" reversible="unspecified">
      <Expression>
        (k14a*MST2a*LATS1/(Km14a+LATS1)+k14b*Ma_F1A*LATS1/(Km14b+LATS1))/default_compartment
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_488" name="Km14a" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_489" name="Km14b" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_490" name="LATS1" order="2" role="substrate"/>
        <ParameterDescription key="FunctionParameter_491" name="MST2a" order="3" role="modifier"/>
        <ParameterDescription key="FunctionParameter_492" name="Ma_F1A" order="4" role="modifier"/>
        <ParameterDescription key="FunctionParameter_493" name="default_compartment" order="5" role="volume"/>
        <ParameterDescription key="FunctionParameter_494" name="k14a" order="6" role="constant"/>
        <ParameterDescription key="FunctionParameter_495" name="k14b" order="7" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_68" name="Function for v15" type="UserDefined" reversible="unspecified">
      <Expression>
        V15*LATS1a/(Km15+LATS1a)/default_compartment
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_408" name="Km15" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_438" name="LATS1a" order="1" role="substrate"/>
        <ParameterDescription key="FunctionParameter_486" name="V15" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_461" name="default_compartment" order="3" role="volume"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_69" name="Function for v16a" type="UserDefined" reversible="unspecified">
      <Expression>
        k16af*Raf_1a*MEK/default_compartment
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_508" name="MEK" order="0" role="substrate"/>
        <ParameterDescription key="FunctionParameter_509" name="Raf_1a" order="1" role="substrate"/>
        <ParameterDescription key="FunctionParameter_510" name="default_compartment" order="2" role="volume"/>
        <ParameterDescription key="FunctionParameter_511" name="k16af" order="3" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_70" name="Function for v16aa" type="UserDefined" reversible="unspecified">
      <Expression>
        k16ar*Ra_Mk/default_compartment
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_475" name="Ra_Mk" order="0" role="substrate"/>
        <ParameterDescription key="FunctionParameter_516" name="default_compartment" order="1" role="volume"/>
        <ParameterDescription key="FunctionParameter_517" name="k16ar" order="2" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_71" name="Function for v16b" type="UserDefined" reversible="unspecified">
      <Expression>
        k16b*Ra_Mk/default_compartment
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_521" name="Ra_Mk" order="0" role="substrate"/>
        <ParameterDescription key="FunctionParameter_522" name="default_compartment" order="1" role="volume"/>
        <ParameterDescription key="FunctionParameter_523" name="k16b" order="2" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_72" name="Function for v17" type="UserDefined" reversible="unspecified">
      <Expression>
        V17*pMEK/(Km17+pMEK+ppMEK*(Km17/Km19))/default_compartment
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_530" name="Km17" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_531" name="Km19" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_532" name="V17" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_533" name="default_compartment" order="3" role="volume"/>
        <ParameterDescription key="FunctionParameter_534" name="pMEK" order="4" role="substrate"/>
        <ParameterDescription key="FunctionParameter_535" name="ppMEK" order="5" role="modifier"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_73" name="Function for v18a" type="UserDefined" reversible="unspecified">
      <Expression>
        k18af*Raf_1a*pMEK/default_compartment
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_528" name="Raf_1a" order="0" role="substrate"/>
        <ParameterDescription key="FunctionParameter_484" name="default_compartment" order="1" role="volume"/>
        <ParameterDescription key="FunctionParameter_542" name="k18af" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_543" name="pMEK" order="3" role="substrate"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_74" name="Function for v18aa" type="UserDefined" reversible="unspecified">
      <Expression>
        k18ar*Ra_pMk/default_compartment
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_487" name="Ra_pMk" order="0" role="substrate"/>
        <ParameterDescription key="FunctionParameter_548" name="default_compartment" order="1" role="volume"/>
        <ParameterDescription key="FunctionParameter_549" name="k18ar" order="2" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_75" name="Function for v18b" type="UserDefined" reversible="unspecified">
      <Expression>
        k18b*Ra_pMk/default_compartment
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_553" name="Ra_pMk" order="0" role="substrate"/>
        <ParameterDescription key="FunctionParameter_554" name="default_compartment" order="1" role="volume"/>
        <ParameterDescription key="FunctionParameter_555" name="k18b" order="2" role="constant"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_76" name="Function for v19" type="UserDefined" reversible="true">
      <Expression>
        V19*ppMEK/(Km19+ppMEK+pMEK*(Km19/Km17))/default_compartment
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_562" name="Km17" order="0" role="constant"/>
        <ParameterDescription key="FunctionParameter_563" name="Km19" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_564" name="V19" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_565" name="default_compartment" order="3" role="volume"/>
        <ParameterDescription key="FunctionParameter_566" name="pMEK" order="4" role="product"/>
        <ParameterDescription key="FunctionParameter_567" name="ppMEK" order="5" role="substrate"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_77" name="Function for v20" type="UserDefined" reversible="true">
      <Expression>
        k20*ppMEK*ERK/(Km20+ERK+pERK*(Km20/Km22))/default_compartment
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_575" name="ERK" order="0" role="substrate"/>
        <ParameterDescription key="FunctionParameter_576" name="Km20" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_577" name="Km22" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_578" name="default_compartment" order="3" role="volume"/>
        <ParameterDescription key="FunctionParameter_579" name="k20" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_580" name="pERK" order="5" role="product"/>
        <ParameterDescription key="FunctionParameter_581" name="ppMEK" order="6" role="modifier"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_78" name="Function for v21" type="UserDefined" reversible="true">
      <MiriamAnnotation>
<rdf:RDF
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Function_78">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-06T08:53:13Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
  </rdf:Description>
</rdf:RDF>

      </MiriamAnnotation>
      <Expression>
        V21*pERK/(Km21+pERK+ppERK*(Km21/Km23)+ERK*(Km21/Ki))/default_compartment
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_590" name="ERK" order="0" role="product"/>
        <ParameterDescription key="FunctionParameter_591" name="Ki" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_592" name="Km21" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_593" name="Km23" order="3" role="constant"/>
        <ParameterDescription key="FunctionParameter_594" name="V21" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_595" name="default_compartment" order="5" role="volume"/>
        <ParameterDescription key="FunctionParameter_596" name="pERK" order="6" role="substrate"/>
        <ParameterDescription key="FunctionParameter_597" name="ppERK" order="7" role="modifier"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_79" name="Function for v22" type="UserDefined" reversible="unspecified">
      <Expression>
        k22*ppMEK*pERK/(Km22+pERK+ERK*(Km22/Km20))/default_compartment
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_529" name="ERK" order="0" role="modifier"/>
        <ParameterDescription key="FunctionParameter_606" name="Km20" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_607" name="Km22" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_608" name="default_compartment" order="3" role="volume"/>
        <ParameterDescription key="FunctionParameter_609" name="k22" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_610" name="pERK" order="5" role="substrate"/>
        <ParameterDescription key="FunctionParameter_611" name="ppMEK" order="6" role="modifier"/>
      </ListOfParameterDescriptions>
    </Function>
    <Function key="Function_80" name="Function for v23" type="UserDefined" reversible="true">
      <Expression>
        V23*ppERK/(Km23+ppERK+pERK*(Km23/Km21)+ERK*(Km23/Ki))/default_compartment
      </Expression>
      <ListOfParameterDescriptions>
        <ParameterDescription key="FunctionParameter_620" name="ERK" order="0" role="modifier"/>
        <ParameterDescription key="FunctionParameter_621" name="Ki" order="1" role="constant"/>
        <ParameterDescription key="FunctionParameter_622" name="Km21" order="2" role="constant"/>
        <ParameterDescription key="FunctionParameter_623" name="Km23" order="3" role="constant"/>
        <ParameterDescription key="FunctionParameter_624" name="V23" order="4" role="constant"/>
        <ParameterDescription key="FunctionParameter_625" name="default_compartment" order="5" role="volume"/>
        <ParameterDescription key="FunctionParameter_626" name="pERK" order="6" role="product"/>
        <ParameterDescription key="FunctionParameter_627" name="ppERK" order="7" role="substrate"/>
      </ListOfParameterDescriptions>
    </Function>
  </ListOfFunctions>
  <Model key="Model_1" name="Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk" simulationType="time" timeUnit="s" volumeUnit="l" areaUnit="m²" lengthUnit="m" quantityUnit="mol" type="deterministic" avogadroConstant="6.0221408570000002e+23">
    <MiriamAnnotation>
<rdf:RDF
   xmlns:CopasiMT="http://www.copasi.org/RDF/MiriamTerms#"
   xmlns:dcterms="http://purl.org/dc/terms/"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="#Model_1">
    <dcterms:created>
      <rdf:Description>
        <dcterms:W3CDTF>2018-04-06T08:52:11Z</dcterms:W3CDTF>
      </rdf:Description>
    </dcterms:created>
    <CopasiMT:is>
      <rdf:Bag>
        <rdf:li rdf:resource="http://identifiers.org/biomodels.db/MODEL1506070001"/>
      </rdf:Bag>
    </CopasiMT:is>
  </rdf:Description>
</rdf:RDF>

    </MiriamAnnotation>
    <Comment>
      
  <body xmlns="http://www.w3.org/1999/xhtml">
    <p> Originally created by libAntimony v1.3 (using libSBML 4.1.0-b2) </p>
  </body>

    </Comment>
    <ListOfCompartments>
      <Compartment key="Compartment_3" name="default_compartment" simulationType="fixed" dimensionality="3">
        <MiriamAnnotation>
<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#' xmlns:dc='http://purl.org/dc/elements/1.1/' xmlns:dcterms='http://purl.org/dc/terms/' xmlns:vCard='http://www.w3.org/2001/vcard-rdf/3.0#' xmlns:bqbiol='http://biomodels.net/biology-qualifiers/' xmlns:bqmodel='http://biomodels.net/model-qualifiers/'>  <rdf:Description rdf:about='#Compartment_3'>
    <bqmodel:is>
      <rdf:Bag>
        <rdf:li rdf:resource='http://identifiers.org/biomodels.sbo/SBO:0000410' />
      </rdf:Bag>
    </bqmodel:is>
  </rdf:Description>
</rdf:RDF>
        </MiriamAnnotation>
      </Compartment>
    </ListOfCompartments>
    <ListOfMetabolites>
      <Metabolite key="Metabolite_7" name="pRaf_1i" simulationType="reactions" compartment="Compartment_3">
      </Metabolite>
      <Metabolite key="Metabolite_8" name="Raf_1" simulationType="reactions" compartment="Compartment_3">
      </Metabolite>
      <Metabolite key="Metabolite_9" name="LATS1a" simulationType="reactions" compartment="Compartment_3">
      </Metabolite>
      <Metabolite key="Metabolite_10" name="Raf_1a" simulationType="reactions" compartment="Compartment_3">
      </Metabolite>
      <Metabolite key="Metabolite_11" name="ppERK" simulationType="reactions" compartment="Compartment_3">
      </Metabolite>
      <Metabolite key="Metabolite_12" name="pMST2i" simulationType="reactions" compartment="Compartment_3">
      </Metabolite>
      <Metabolite key="Metabolite_13" name="pMi_pRi" simulationType="reactions" compartment="Compartment_3">
      </Metabolite>
      <Metabolite key="Metabolite_14" name="MST2" simulationType="reactions" compartment="Compartment_3">
      </Metabolite>
      <Metabolite key="Metabolite_15" name="M_pRi" simulationType="reactions" compartment="Compartment_3">
      </Metabolite>
      <Metabolite key="Metabolite_16" name="MST2a" simulationType="reactions" compartment="Compartment_3">
      </Metabolite>
      <Metabolite key="Metabolite_17" name="F1A" simulationType="reactions" compartment="Compartment_3">
      </Metabolite>
      <Metabolite key="Metabolite_18" name="Ma_F1A" simulationType="reactions" compartment="Compartment_3">
      </Metabolite>
      <Metabolite key="Metabolite_19" name="M_F1A" simulationType="reactions" compartment="Compartment_3">
      </Metabolite>
      <Metabolite key="Metabolite_20" name="LATS1" simulationType="reactions" compartment="Compartment_3">
      </Metabolite>
      <Metabolite key="Metabolite_21" name="MEK" simulationType="reactions" compartment="Compartment_3">
      </Metabolite>
      <Metabolite key="Metabolite_22" name="Ra_Mk" simulationType="reactions" compartment="Compartment_3">
      </Metabolite>
      <Metabolite key="Metabolite_23" name="pMEK" simulationType="reactions" compartment="Compartment_3">
      </Metabolite>
      <Metabolite key="Metabolite_24" name="ppMEK" simulationType="reactions" compartment="Compartment_3">
      </Metabolite>
      <Metabolite key="Metabolite_25" name="Ra_pMk" simulationType="reactions" compartment="Compartment_3">
      </Metabolite>
      <Metabolite key="Metabolite_26" name="ERK" simulationType="reactions" compartment="Compartment_3">
      </Metabolite>
      <Metabolite key="Metabolite_27" name="pERK" simulationType="reactions" compartment="Compartment_3">
      </Metabolite>
    </ListOfMetabolites>
    <ListOfModelValues>
      <ModelValue key="ModelValue_0" name="k1" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_1" name="RasGTP" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_2" name="Km1" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_3" name="k2a" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_4" name="Km2a" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_5" name="k2b" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_6" name="Kin" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_7" name="Km2b" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_8" name="V3" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_9" name="Km3" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_10" name="Fa" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_11" name="Ka" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_12" name="V4" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_13" name="Km4" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_14" name="k5f" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_15" name="k5r" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_16" name="k6f" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_17" name="k6r" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_18" name="k7" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_19" name="AKTa" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_20" name="Kact" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_21" name="Km7" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_22" name="k8" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_23" name="PP2A" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_24" name="Km8" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_25" name="k9" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_26" name="k10" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_27" name="Km10" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_28" name="k11f" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_29" name="k11r" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_30" name="k12f" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_31" name="k12r" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_32" name="V13" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_33" name="Km13" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_34" name="k14a" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_35" name="Km14a" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_36" name="k14b" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_37" name="Km14b" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_38" name="V15" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_39" name="Km15" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_40" name="k16af" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_41" name="k16ar" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_42" name="k16b" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_43" name="V17" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_44" name="Km17" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_45" name="Km19" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_46" name="k18af" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_47" name="k18ar" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_48" name="k18b" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_49" name="V19" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_50" name="k20" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_51" name="Km20" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_52" name="Km22" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_53" name="V21" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_54" name="Km21" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_55" name="Km23" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_56" name="Ki" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_57" name="k22" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_58" name="V23" simulationType="fixed">
      </ModelValue>
      <ModelValue key="ModelValue_59" name="pMST2" simulationType="fixed">
      </ModelValue>
    </ListOfModelValues>
    <ListOfReactions>
      <Reaction key="Reaction_10" name="v1" reversible="true" fast="false">
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_7" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_8" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_7" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4825" name="Km1" value="1"/>
          <Constant key="Parameter_4857" name="RasGTP" value="50"/>
          <Constant key="Parameter_4858" name="k1" value="0.02"/>
        </ListOfConstants>
        <KineticLaw function="Function_50" unitType="Default" scalingCompartment="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_333">
              <SourceParameter reference="ModelValue_2"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_334">
              <SourceParameter reference="ModelValue_1"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_335">
              <SourceParameter reference="Compartment_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_336">
              <SourceParameter reference="ModelValue_0"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_337">
              <SourceParameter reference="Metabolite_7"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_11" name="v2" reversible="true" fast="false">
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_8" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_7" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_9" stoichiometry="2"/>
          <Modifier metabolite="Metabolite_8" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4859" name="Kin" value="20"/>
          <Constant key="Parameter_4860" name="Km2a" value="1"/>
          <Constant key="Parameter_4861" name="Km2b" value="1"/>
          <Constant key="Parameter_4862" name="k2a" value="0.01"/>
          <Constant key="Parameter_4863" name="k2b" value="0.01"/>
        </ListOfConstants>
        <KineticLaw function="Function_51" unitType="Default" scalingCompartment="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_346">
              <SourceParameter reference="ModelValue_6"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_347">
              <SourceParameter reference="ModelValue_4"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_348">
              <SourceParameter reference="ModelValue_7"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_349">
              <SourceParameter reference="Metabolite_9"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_350">
              <SourceParameter reference="Metabolite_8"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_351">
              <SourceParameter reference="Compartment_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_352">
              <SourceParameter reference="ModelValue_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_353">
              <SourceParameter reference="ModelValue_5"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_12" name="v3" reversible="true" fast="false">
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_8" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_10" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_11" stoichiometry="3"/>
          <Modifier metabolite="Metabolite_8" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4864" name="Fa" value="0.5"/>
          <Constant key="Parameter_4852" name="Ka" value="1000"/>
          <Constant key="Parameter_4853" name="Km3" value="1"/>
          <Constant key="Parameter_4854" name="V3" value="2"/>
        </ListOfConstants>
        <KineticLaw function="Function_52" unitType="Default" scalingCompartment="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_343">
              <SourceParameter reference="ModelValue_10"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_362">
              <SourceParameter reference="ModelValue_11"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_363">
              <SourceParameter reference="ModelValue_9"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_364">
              <SourceParameter reference="Metabolite_8"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_365">
              <SourceParameter reference="ModelValue_8"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_366">
              <SourceParameter reference="Compartment_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_367">
              <SourceParameter reference="Metabolite_11"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_13" name="v4" reversible="true" fast="false">
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_10" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_8" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_10" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4855" name="Km4" value="1"/>
          <Constant key="Parameter_4856" name="V4" value="1"/>
        </ListOfConstants>
        <KineticLaw function="Function_53" unitType="Default" scalingCompartment="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_345">
              <SourceParameter reference="ModelValue_13"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_344">
              <SourceParameter reference="Metabolite_10"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_332">
              <SourceParameter reference="ModelValue_12"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_375">
              <SourceParameter reference="Compartment_3"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_14" name="v5a" reversible="true" fast="false">
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_12" stoichiometry="1"/>
          <Substrate metabolite="Metabolite_7" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_13" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_12" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_7" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4826" name="k5f" value="0.01"/>
        </ListOfConstants>
        <KineticLaw function="Function_54" unitType="Default" scalingCompartment="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_380">
              <SourceParameter reference="Compartment_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_381">
              <SourceParameter reference="ModelValue_14"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_382">
              <SourceParameter reference="Metabolite_12"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_383">
              <SourceParameter reference="Metabolite_7"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_15" name="v5b" reversible="true" fast="false">
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_13" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_12" stoichiometry="1"/>
          <Product metabolite="Metabolite_7" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_13" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4827" name="k5r" value="0.1"/>
        </ListOfConstants>
        <KineticLaw function="Function_55" unitType="Default" scalingCompartment="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_246">
              <SourceParameter reference="Compartment_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_388">
              <SourceParameter reference="ModelValue_15"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_389">
              <SourceParameter reference="Metabolite_13"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_16" name="v6a" reversible="true" fast="false">
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_14" stoichiometry="1"/>
          <Substrate metabolite="Metabolite_7" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_15" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_14" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_7" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4828" name="k6f" value="0.001"/>
        </ListOfConstants>
        <KineticLaw function="Function_56" unitType="Default" scalingCompartment="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_394">
              <SourceParameter reference="Metabolite_14"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_395">
              <SourceParameter reference="Compartment_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_396">
              <SourceParameter reference="ModelValue_16"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_397">
              <SourceParameter reference="Metabolite_7"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_17" name="v6b" reversible="true" fast="false">
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_15" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_14" stoichiometry="1"/>
          <Product metabolite="Metabolite_7" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_15" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4829" name="k6r" value="0.1"/>
        </ListOfConstants>
        <KineticLaw function="Function_57" unitType="Default" scalingCompartment="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_315">
              <SourceParameter reference="Metabolite_15"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_402">
              <SourceParameter reference="Compartment_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_403">
              <SourceParameter reference="ModelValue_17"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_18" name="v7" reversible="true" fast="false">
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_14" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_12" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_14" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4830" name="AKTa" value="20"/>
          <Constant key="Parameter_4831" name="Kact" value="0.2"/>
          <Constant key="Parameter_4832" name="Km7" value="1"/>
          <Constant key="Parameter_4833" name="RasGTP" value="50"/>
          <Constant key="Parameter_4834" name="k7" value="0.008"/>
        </ListOfConstants>
        <KineticLaw function="Function_58" unitType="Default" scalingCompartment="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_411">
              <SourceParameter reference="ModelValue_19"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_412">
              <SourceParameter reference="ModelValue_20"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_413">
              <SourceParameter reference="ModelValue_21"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_414">
              <SourceParameter reference="Metabolite_14"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_415">
              <SourceParameter reference="ModelValue_1"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_416">
              <SourceParameter reference="Compartment_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_417">
              <SourceParameter reference="ModelValue_18"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_19" name="v8" reversible="true" fast="false">
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_12" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_14" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_12" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4835" name="Km8" value="1"/>
          <Constant key="Parameter_4836" name="PP2A" value="50"/>
          <Constant key="Parameter_4837" name="k8" value="0.01"/>
        </ListOfConstants>
        <KineticLaw function="Function_59" unitType="Default" scalingCompartment="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_407">
              <SourceParameter reference="ModelValue_24"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_393">
              <SourceParameter reference="ModelValue_23"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_425">
              <SourceParameter reference="Compartment_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_426">
              <SourceParameter reference="ModelValue_22"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_427">
              <SourceParameter reference="Metabolite_12"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_20" name="v9" reversible="true" fast="false">
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_14" stoichiometry="2"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_16" stoichiometry="2"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_14" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4838" name="k9" value="0.00035"/>
        </ListOfConstants>
        <KineticLaw function="Function_60" unitType="Default" scalingCompartment="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_410">
              <SourceParameter reference="Metabolite_14"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_264">
              <SourceParameter reference="Compartment_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_433">
              <SourceParameter reference="ModelValue_25"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_21" name="v10" reversible="true" fast="false">
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_16" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_14" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_16" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4839" name="Km10" value="50"/>
          <Constant key="Parameter_4840" name="PP2A" value="50"/>
          <Constant key="Parameter_4841" name="k10" value="10"/>
        </ListOfConstants>
        <KineticLaw function="Function_61" unitType="Default" scalingCompartment="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_439">
              <SourceParameter reference="ModelValue_27"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_440">
              <SourceParameter reference="Metabolite_16"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_441">
              <SourceParameter reference="ModelValue_23"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_442">
              <SourceParameter reference="Compartment_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_443">
              <SourceParameter reference="ModelValue_26"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_22" name="v11a" reversible="true" fast="false">
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_16" stoichiometry="1"/>
          <Substrate metabolite="Metabolite_17" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_18" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_16" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_17" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4842" name="k11f" value="0.01"/>
        </ListOfConstants>
        <KineticLaw function="Function_62" unitType="Default" scalingCompartment="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_437">
              <SourceParameter reference="Metabolite_17"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_449">
              <SourceParameter reference="Metabolite_16"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_450">
              <SourceParameter reference="Compartment_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_451">
              <SourceParameter reference="ModelValue_28"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_23" name="v11b" reversible="true" fast="false">
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_18" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_16" stoichiometry="1"/>
          <Product metabolite="Metabolite_17" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_18" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4843" name="k11r" value="0.1"/>
        </ListOfConstants>
        <KineticLaw function="Function_63" unitType="Default" scalingCompartment="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_409">
              <SourceParameter reference="Metabolite_18"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_456">
              <SourceParameter reference="Compartment_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_457">
              <SourceParameter reference="ModelValue_29"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_24" name="v12a" reversible="true" fast="false">
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_14" stoichiometry="1"/>
          <Substrate metabolite="Metabolite_17" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_19" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_14" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_17" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4844" name="k12f" value="0.01"/>
        </ListOfConstants>
        <KineticLaw function="Function_64" unitType="Default" scalingCompartment="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_462">
              <SourceParameter reference="Metabolite_17"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_463">
              <SourceParameter reference="Metabolite_14"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_464">
              <SourceParameter reference="Compartment_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_465">
              <SourceParameter reference="ModelValue_30"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_25" name="v12b" reversible="true" fast="false">
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_19" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_14" stoichiometry="1"/>
          <Product metabolite="Metabolite_17" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_19" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4845" name="k12r" value="1"/>
        </ListOfConstants>
        <KineticLaw function="Function_65" unitType="Default" scalingCompartment="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_262">
              <SourceParameter reference="Metabolite_19"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_470">
              <SourceParameter reference="Compartment_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_471">
              <SourceParameter reference="ModelValue_31"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_26" name="v13" reversible="true" fast="false">
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_19" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_18" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_19" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4846" name="Km13" value="50"/>
          <Constant key="Parameter_4847" name="V13" value="10"/>
        </ListOfConstants>
        <KineticLaw function="Function_66" unitType="Default" scalingCompartment="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_476">
              <SourceParameter reference="ModelValue_33"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_477">
              <SourceParameter reference="Metabolite_19"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_478">
              <SourceParameter reference="ModelValue_32"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_479">
              <SourceParameter reference="Compartment_3"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_27" name="v14" reversible="true" fast="false">
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_20" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_9" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_16" stoichiometry="2"/>
          <Modifier metabolite="Metabolite_18" stoichiometry="2"/>
          <Modifier metabolite="Metabolite_20" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4848" name="Km14a" value="50"/>
          <Constant key="Parameter_4849" name="Km14b" value="50"/>
          <Constant key="Parameter_4850" name="k14a" value="0.05"/>
          <Constant key="Parameter_4851" name="k14b" value="0.05"/>
        </ListOfConstants>
        <KineticLaw function="Function_67" unitType="Default" scalingCompartment="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_488">
              <SourceParameter reference="ModelValue_35"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_489">
              <SourceParameter reference="ModelValue_37"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_490">
              <SourceParameter reference="Metabolite_20"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_491">
              <SourceParameter reference="Metabolite_16"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_492">
              <SourceParameter reference="Metabolite_18"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_493">
              <SourceParameter reference="Compartment_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_494">
              <SourceParameter reference="ModelValue_34"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_495">
              <SourceParameter reference="ModelValue_36"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_28" name="v15" reversible="true" fast="false">
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_9" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_20" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_9" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4865" name="Km15" value="50"/>
          <Constant key="Parameter_4866" name="V15" value="0.05"/>
        </ListOfConstants>
        <KineticLaw function="Function_68" unitType="Default" scalingCompartment="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_408">
              <SourceParameter reference="ModelValue_39"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_438">
              <SourceParameter reference="Metabolite_9"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_486">
              <SourceParameter reference="ModelValue_38"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_461">
              <SourceParameter reference="Compartment_3"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_29" name="v16a" reversible="true" fast="false">
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_10" stoichiometry="1"/>
          <Substrate metabolite="Metabolite_21" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_22" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_10" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_21" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4867" name="k16af" value="0.01"/>
        </ListOfConstants>
        <KineticLaw function="Function_69" unitType="Default" scalingCompartment="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_508">
              <SourceParameter reference="Metabolite_21"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_509">
              <SourceParameter reference="Metabolite_10"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_510">
              <SourceParameter reference="Compartment_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_511">
              <SourceParameter reference="ModelValue_40"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_30" name="v16aa" reversible="true" fast="false">
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_22" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_10" stoichiometry="1"/>
          <Product metabolite="Metabolite_21" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_22" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4868" name="k16ar" value="1"/>
        </ListOfConstants>
        <KineticLaw function="Function_70" unitType="Default" scalingCompartment="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_475">
              <SourceParameter reference="Metabolite_22"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_516">
              <SourceParameter reference="Compartment_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_517">
              <SourceParameter reference="ModelValue_41"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_31" name="v16b" reversible="true" fast="false">
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_22" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_23" stoichiometry="1"/>
          <Product metabolite="Metabolite_10" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_22" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4869" name="k16b" value="5"/>
        </ListOfConstants>
        <KineticLaw function="Function_71" unitType="Default" scalingCompartment="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_521">
              <SourceParameter reference="Metabolite_22"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_522">
              <SourceParameter reference="Compartment_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_523">
              <SourceParameter reference="ModelValue_42"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_32" name="v17" reversible="true" fast="false">
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_23" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_21" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_24" stoichiometry="2"/>
          <Modifier metabolite="Metabolite_23" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4870" name="Km17" value="400"/>
          <Constant key="Parameter_4871" name="Km19" value="400"/>
          <Constant key="Parameter_4872" name="V17" value="250"/>
        </ListOfConstants>
        <KineticLaw function="Function_72" unitType="Default" scalingCompartment="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_530">
              <SourceParameter reference="ModelValue_44"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_531">
              <SourceParameter reference="ModelValue_45"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_532">
              <SourceParameter reference="ModelValue_43"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_533">
              <SourceParameter reference="Compartment_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_534">
              <SourceParameter reference="Metabolite_23"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_535">
              <SourceParameter reference="Metabolite_24"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_33" name="v18a" reversible="true" fast="false">
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_10" stoichiometry="1"/>
          <Substrate metabolite="Metabolite_23" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_25" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_10" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_23" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4873" name="k18af" value="0.01"/>
        </ListOfConstants>
        <KineticLaw function="Function_73" unitType="Default" scalingCompartment="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_528">
              <SourceParameter reference="Metabolite_10"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_484">
              <SourceParameter reference="Compartment_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_542">
              <SourceParameter reference="ModelValue_46"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_543">
              <SourceParameter reference="Metabolite_23"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_34" name="v18aa" reversible="true" fast="false">
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_25" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_10" stoichiometry="1"/>
          <Product metabolite="Metabolite_23" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_25" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4874" name="k18ar" value="1"/>
        </ListOfConstants>
        <KineticLaw function="Function_74" unitType="Default" scalingCompartment="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_487">
              <SourceParameter reference="Metabolite_25"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_548">
              <SourceParameter reference="Compartment_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_549">
              <SourceParameter reference="ModelValue_47"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_35" name="v18b" reversible="true" fast="false">
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_25" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_24" stoichiometry="1"/>
          <Product metabolite="Metabolite_10" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_25" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4875" name="k18b" value="5"/>
        </ListOfConstants>
        <KineticLaw function="Function_75" unitType="Default" scalingCompartment="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_553">
              <SourceParameter reference="Metabolite_25"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_554">
              <SourceParameter reference="Compartment_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_555">
              <SourceParameter reference="ModelValue_48"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_36" name="v19" reversible="true" fast="false">
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_24" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_23" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_24" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_23" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4876" name="Km17" value="400"/>
          <Constant key="Parameter_4877" name="Km19" value="400"/>
          <Constant key="Parameter_4878" name="V19" value="250"/>
        </ListOfConstants>
        <KineticLaw function="Function_76" unitType="Default" scalingCompartment="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_562">
              <SourceParameter reference="ModelValue_44"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_563">
              <SourceParameter reference="ModelValue_45"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_564">
              <SourceParameter reference="ModelValue_49"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_565">
              <SourceParameter reference="Compartment_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_566">
              <SourceParameter reference="Metabolite_23"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_567">
              <SourceParameter reference="Metabolite_24"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_37" name="v20" reversible="true" fast="false">
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_26" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_27" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_24" stoichiometry="2"/>
          <Modifier metabolite="Metabolite_26" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_27" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4879" name="Km20" value="500"/>
          <Constant key="Parameter_4880" name="Km22" value="500"/>
          <Constant key="Parameter_4881" name="k20" value="1"/>
        </ListOfConstants>
        <KineticLaw function="Function_77" unitType="Default" scalingCompartment="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_575">
              <SourceParameter reference="Metabolite_26"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_576">
              <SourceParameter reference="ModelValue_51"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_577">
              <SourceParameter reference="ModelValue_52"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_578">
              <SourceParameter reference="Compartment_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_579">
              <SourceParameter reference="ModelValue_50"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_580">
              <SourceParameter reference="Metabolite_27"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_581">
              <SourceParameter reference="Metabolite_24"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_38" name="v21" reversible="true" fast="false">
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_27" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_26" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_11" stoichiometry="2"/>
          <Modifier metabolite="Metabolite_27" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_26" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4882" name="Ki" value="1"/>
          <Constant key="Parameter_4883" name="Km21" value="500"/>
          <Constant key="Parameter_4884" name="Km23" value="100"/>
          <Constant key="Parameter_4885" name="V21" value="100"/>
        </ListOfConstants>
        <KineticLaw function="Function_78" unitType="Default" scalingCompartment="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_590">
              <SourceParameter reference="Metabolite_26"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_591">
              <SourceParameter reference="ModelValue_56"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_592">
              <SourceParameter reference="ModelValue_54"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_593">
              <SourceParameter reference="ModelValue_55"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_594">
              <SourceParameter reference="ModelValue_53"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_595">
              <SourceParameter reference="Compartment_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_596">
              <SourceParameter reference="Metabolite_27"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_597">
              <SourceParameter reference="Metabolite_11"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_39" name="v22" reversible="true" fast="false">
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_27" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_11" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_24" stoichiometry="2"/>
          <Modifier metabolite="Metabolite_26" stoichiometry="2"/>
          <Modifier metabolite="Metabolite_27" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4886" name="Km20" value="500"/>
          <Constant key="Parameter_4887" name="Km22" value="500"/>
          <Constant key="Parameter_4888" name="k22" value="10"/>
        </ListOfConstants>
        <KineticLaw function="Function_79" unitType="Default" scalingCompartment="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_529">
              <SourceParameter reference="Metabolite_26"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_606">
              <SourceParameter reference="ModelValue_51"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_607">
              <SourceParameter reference="ModelValue_52"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_608">
              <SourceParameter reference="Compartment_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_609">
              <SourceParameter reference="ModelValue_57"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_610">
              <SourceParameter reference="Metabolite_27"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_611">
              <SourceParameter reference="Metabolite_24"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
      <Reaction key="Reaction_40" name="v23" reversible="true" fast="false">
        <ListOfSubstrates>
          <Substrate metabolite="Metabolite_11" stoichiometry="1"/>
        </ListOfSubstrates>
        <ListOfProducts>
          <Product metabolite="Metabolite_27" stoichiometry="1"/>
        </ListOfProducts>
        <ListOfModifiers>
          <Modifier metabolite="Metabolite_26" stoichiometry="2"/>
          <Modifier metabolite="Metabolite_11" stoichiometry="1"/>
          <Modifier metabolite="Metabolite_27" stoichiometry="1"/>
        </ListOfModifiers>
        <ListOfConstants>
          <Constant key="Parameter_4889" name="Ki" value="1"/>
          <Constant key="Parameter_4890" name="Km21" value="500"/>
          <Constant key="Parameter_4891" name="Km23" value="100"/>
          <Constant key="Parameter_4892" name="V23" value="100"/>
        </ListOfConstants>
        <KineticLaw function="Function_80" unitType="Default" scalingCompartment="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]">
          <ListOfCallParameters>
            <CallParameter functionParameter="FunctionParameter_620">
              <SourceParameter reference="Metabolite_26"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_621">
              <SourceParameter reference="ModelValue_56"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_622">
              <SourceParameter reference="ModelValue_54"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_623">
              <SourceParameter reference="ModelValue_55"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_624">
              <SourceParameter reference="ModelValue_58"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_625">
              <SourceParameter reference="Compartment_3"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_626">
              <SourceParameter reference="Metabolite_27"/>
            </CallParameter>
            <CallParameter functionParameter="FunctionParameter_627">
              <SourceParameter reference="Metabolite_11"/>
            </CallParameter>
          </ListOfCallParameters>
        </KineticLaw>
      </Reaction>
    </ListOfReactions>
    <ListOfModelParameterSets activeSet="ModelParameterSet_1">
      <ModelParameterSet key="ModelParameterSet_1" name="Initial State">
        <ModelParameterGroup cn="String=Initial Time" type="Group">
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk" value="0" type="Model" simulationType="time"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Initial Compartment Sizes" type="Group">
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment]" value="1" type="Compartment" simulationType="fixed"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Initial Species Values" type="Group">
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment],Vector=Metabolites[pRaf_1i]" value="4.5166056427500002e+26" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment],Vector=Metabolites[Raf_1]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment],Vector=Metabolites[LATS1a]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment],Vector=Metabolites[Raf_1a]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment],Vector=Metabolites[ppERK]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment],Vector=Metabolites[pMST2i]" value="6.0221408570000002e+23" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment],Vector=Metabolites[pMi_pRi]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment],Vector=Metabolites[MST2]" value="9.0332112855000003e+26" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment],Vector=Metabolites[M_pRi]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment],Vector=Metabolites[MST2a]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment],Vector=Metabolites[F1A]" value="6.0221408569999997e+25" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment],Vector=Metabolites[Ma_F1A]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment],Vector=Metabolites[M_F1A]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment],Vector=Metabolites[LATS1]" value="6.0221408569999997e+25" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment],Vector=Metabolites[MEK]" value="9.6354253711999996e+26" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment],Vector=Metabolites[Ra_Mk]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment],Vector=Metabolites[pMEK]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment],Vector=Metabolites[ppMEK]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment],Vector=Metabolites[Ra_pMk]" value="0" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment],Vector=Metabolites[ERK]" value="1.8066422571000001e+27" type="Species" simulationType="reactions"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment],Vector=Metabolites[pERK]" value="0" type="Species" simulationType="reactions"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Initial Global Quantities" type="Group">
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k1]" value="0.02" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[RasGTP]" value="50" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km1]" value="1" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k2a]" value="0.01" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km2a]" value="1" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k2b]" value="0.01" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Kin]" value="20" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km2b]" value="1" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[V3]" value="2" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km3]" value="1" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Fa]" value="0.5" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Ka]" value="1000" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[V4]" value="1" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km4]" value="1" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k5f]" value="0.01" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k5r]" value="0.10000000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k6f]" value="0.001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k6r]" value="0.10000000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k7]" value="0.0080000000000000002" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[AKTa]" value="20" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Kact]" value="0.20000000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km7]" value="1" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k8]" value="0.01" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[PP2A]" value="50" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km8]" value="1" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k9]" value="0.00035" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k10]" value="10" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km10]" value="50" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k11f]" value="0.01" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k11r]" value="0.10000000000000001" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k12f]" value="0.01" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k12r]" value="1" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[V13]" value="10" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km13]" value="50" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k14a]" value="0.050000000000000003" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km14a]" value="50" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k14b]" value="0.050000000000000003" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km14b]" value="50" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[V15]" value="0.050000000000000003" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km15]" value="50" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k16af]" value="0.01" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k16ar]" value="1" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k16b]" value="5" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[V17]" value="250" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km17]" value="400" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km19]" value="400" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k18af]" value="0.01" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k18ar]" value="1" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k18b]" value="5" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[V19]" value="250" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k20]" value="1" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km20]" value="500" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km22]" value="500" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[V21]" value="100" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km21]" value="500" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km23]" value="100" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Ki]" value="1" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k22]" value="10" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[V23]" value="100" type="ModelValue" simulationType="fixed"/>
          <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[pMST2]" value="0" type="ModelValue" simulationType="fixed"/>
        </ModelParameterGroup>
        <ModelParameterGroup cn="String=Kinetic Parameters" type="Group">
          <ModelParameterGroup cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v1]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v1],ParameterGroup=Parameters,Parameter=Km1" value="1" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km1],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v1],ParameterGroup=Parameters,Parameter=RasGTP" value="50" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[RasGTP],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v1],ParameterGroup=Parameters,Parameter=k1" value="0.02" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k1],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v2]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v2],ParameterGroup=Parameters,Parameter=Kin" value="20" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Kin],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v2],ParameterGroup=Parameters,Parameter=Km2a" value="1" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km2a],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v2],ParameterGroup=Parameters,Parameter=Km2b" value="1" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km2b],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v2],ParameterGroup=Parameters,Parameter=k2a" value="0.01" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k2a],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v2],ParameterGroup=Parameters,Parameter=k2b" value="0.01" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k2b],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v3]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v3],ParameterGroup=Parameters,Parameter=Fa" value="0.5" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Fa],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v3],ParameterGroup=Parameters,Parameter=Ka" value="1000" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Ka],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v3],ParameterGroup=Parameters,Parameter=Km3" value="1" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km3],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v3],ParameterGroup=Parameters,Parameter=V3" value="2" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[V3],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v4]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v4],ParameterGroup=Parameters,Parameter=Km4" value="1" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km4],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v4],ParameterGroup=Parameters,Parameter=V4" value="1" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[V4],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v5a]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v5a],ParameterGroup=Parameters,Parameter=k5f" value="0.01" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k5f],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v5b]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v5b],ParameterGroup=Parameters,Parameter=k5r" value="0.10000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k5r],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v6a]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v6a],ParameterGroup=Parameters,Parameter=k6f" value="0.001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k6f],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v6b]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v6b],ParameterGroup=Parameters,Parameter=k6r" value="0.10000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k6r],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v7]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v7],ParameterGroup=Parameters,Parameter=AKTa" value="20" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[AKTa],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v7],ParameterGroup=Parameters,Parameter=Kact" value="0.20000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Kact],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v7],ParameterGroup=Parameters,Parameter=Km7" value="1" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km7],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v7],ParameterGroup=Parameters,Parameter=RasGTP" value="50" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[RasGTP],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v7],ParameterGroup=Parameters,Parameter=k7" value="0.0080000000000000002" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k7],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v8]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v8],ParameterGroup=Parameters,Parameter=Km8" value="1" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km8],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v8],ParameterGroup=Parameters,Parameter=PP2A" value="50" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[PP2A],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v8],ParameterGroup=Parameters,Parameter=k8" value="0.01" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k8],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v9]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v9],ParameterGroup=Parameters,Parameter=k9" value="0.00035" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k9],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v10]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v10],ParameterGroup=Parameters,Parameter=Km10" value="50" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km10],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v10],ParameterGroup=Parameters,Parameter=PP2A" value="50" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[PP2A],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v10],ParameterGroup=Parameters,Parameter=k10" value="10" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k10],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v11a]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v11a],ParameterGroup=Parameters,Parameter=k11f" value="0.01" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k11f],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v11b]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v11b],ParameterGroup=Parameters,Parameter=k11r" value="0.10000000000000001" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k11r],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v12a]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v12a],ParameterGroup=Parameters,Parameter=k12f" value="0.01" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k12f],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v12b]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v12b],ParameterGroup=Parameters,Parameter=k12r" value="1" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k12r],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v13]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v13],ParameterGroup=Parameters,Parameter=Km13" value="50" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km13],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v13],ParameterGroup=Parameters,Parameter=V13" value="10" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[V13],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v14]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v14],ParameterGroup=Parameters,Parameter=Km14a" value="50" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km14a],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v14],ParameterGroup=Parameters,Parameter=Km14b" value="50" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km14b],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v14],ParameterGroup=Parameters,Parameter=k14a" value="0.050000000000000003" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k14a],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v14],ParameterGroup=Parameters,Parameter=k14b" value="0.050000000000000003" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k14b],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v15]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v15],ParameterGroup=Parameters,Parameter=Km15" value="50" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km15],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v15],ParameterGroup=Parameters,Parameter=V15" value="0.050000000000000003" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[V15],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v16a]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v16a],ParameterGroup=Parameters,Parameter=k16af" value="0.01" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k16af],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v16aa]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v16aa],ParameterGroup=Parameters,Parameter=k16ar" value="1" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k16ar],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v16b]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v16b],ParameterGroup=Parameters,Parameter=k16b" value="5" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k16b],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v17]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v17],ParameterGroup=Parameters,Parameter=Km17" value="400" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km17],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v17],ParameterGroup=Parameters,Parameter=Km19" value="400" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km19],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v17],ParameterGroup=Parameters,Parameter=V17" value="250" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[V17],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v18a]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v18a],ParameterGroup=Parameters,Parameter=k18af" value="0.01" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k18af],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v18aa]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v18aa],ParameterGroup=Parameters,Parameter=k18ar" value="1" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k18ar],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v18b]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v18b],ParameterGroup=Parameters,Parameter=k18b" value="5" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k18b],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v19]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v19],ParameterGroup=Parameters,Parameter=Km17" value="400" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km17],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v19],ParameterGroup=Parameters,Parameter=Km19" value="400" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km19],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v19],ParameterGroup=Parameters,Parameter=V19" value="250" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[V19],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v20]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v20],ParameterGroup=Parameters,Parameter=Km20" value="500" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km20],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v20],ParameterGroup=Parameters,Parameter=Km22" value="500" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km22],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v20],ParameterGroup=Parameters,Parameter=k20" value="1" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k20],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v21]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v21],ParameterGroup=Parameters,Parameter=Ki" value="1" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Ki],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v21],ParameterGroup=Parameters,Parameter=Km21" value="500" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km21],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v21],ParameterGroup=Parameters,Parameter=Km23" value="100" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km23],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v21],ParameterGroup=Parameters,Parameter=V21" value="100" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[V21],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v22]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v22],ParameterGroup=Parameters,Parameter=Km20" value="500" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km20],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v22],ParameterGroup=Parameters,Parameter=Km22" value="500" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km22],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v22],ParameterGroup=Parameters,Parameter=k22" value="10" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[k22],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
          <ModelParameterGroup cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v23]" type="Reaction">
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v23],ParameterGroup=Parameters,Parameter=Ki" value="1" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Ki],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v23],ParameterGroup=Parameters,Parameter=Km21" value="500" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km21],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v23],ParameterGroup=Parameters,Parameter=Km23" value="100" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[Km23],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
            <ModelParameter cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Reactions[v23],ParameterGroup=Parameters,Parameter=V23" value="100" type="ReactionParameter" simulationType="assignment">
              <InitialExpression>
                &lt;CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Values[V23],Reference=InitialValue>
              </InitialExpression>
            </ModelParameter>
          </ModelParameterGroup>
        </ModelParameterGroup>
      </ModelParameterSet>
    </ListOfModelParameterSets>
    <StateTemplate>
      <StateTemplateVariable objectReference="Model_1"/>
      <StateTemplateVariable objectReference="Metabolite_14"/>
      <StateTemplateVariable objectReference="Metabolite_10"/>
      <StateTemplateVariable objectReference="Metabolite_7"/>
      <StateTemplateVariable objectReference="Metabolite_16"/>
      <StateTemplateVariable objectReference="Metabolite_27"/>
      <StateTemplateVariable objectReference="Metabolite_23"/>
      <StateTemplateVariable objectReference="Metabolite_8"/>
      <StateTemplateVariable objectReference="Metabolite_19"/>
      <StateTemplateVariable objectReference="Metabolite_9"/>
      <StateTemplateVariable objectReference="Metabolite_12"/>
      <StateTemplateVariable objectReference="Metabolite_22"/>
      <StateTemplateVariable objectReference="Metabolite_24"/>
      <StateTemplateVariable objectReference="Metabolite_26"/>
      <StateTemplateVariable objectReference="Metabolite_15"/>
      <StateTemplateVariable objectReference="Metabolite_13"/>
      <StateTemplateVariable objectReference="Metabolite_21"/>
      <StateTemplateVariable objectReference="Metabolite_18"/>
      <StateTemplateVariable objectReference="Metabolite_25"/>
      <StateTemplateVariable objectReference="Metabolite_20"/>
      <StateTemplateVariable objectReference="Metabolite_17"/>
      <StateTemplateVariable objectReference="Metabolite_11"/>
      <StateTemplateVariable objectReference="Compartment_3"/>
      <StateTemplateVariable objectReference="ModelValue_0"/>
      <StateTemplateVariable objectReference="ModelValue_1"/>
      <StateTemplateVariable objectReference="ModelValue_2"/>
      <StateTemplateVariable objectReference="ModelValue_3"/>
      <StateTemplateVariable objectReference="ModelValue_4"/>
      <StateTemplateVariable objectReference="ModelValue_5"/>
      <StateTemplateVariable objectReference="ModelValue_6"/>
      <StateTemplateVariable objectReference="ModelValue_7"/>
      <StateTemplateVariable objectReference="ModelValue_8"/>
      <StateTemplateVariable objectReference="ModelValue_9"/>
      <StateTemplateVariable objectReference="ModelValue_10"/>
      <StateTemplateVariable objectReference="ModelValue_11"/>
      <StateTemplateVariable objectReference="ModelValue_12"/>
      <StateTemplateVariable objectReference="ModelValue_13"/>
      <StateTemplateVariable objectReference="ModelValue_14"/>
      <StateTemplateVariable objectReference="ModelValue_15"/>
      <StateTemplateVariable objectReference="ModelValue_16"/>
      <StateTemplateVariable objectReference="ModelValue_17"/>
      <StateTemplateVariable objectReference="ModelValue_18"/>
      <StateTemplateVariable objectReference="ModelValue_19"/>
      <StateTemplateVariable objectReference="ModelValue_20"/>
      <StateTemplateVariable objectReference="ModelValue_21"/>
      <StateTemplateVariable objectReference="ModelValue_22"/>
      <StateTemplateVariable objectReference="ModelValue_23"/>
      <StateTemplateVariable objectReference="ModelValue_24"/>
      <StateTemplateVariable objectReference="ModelValue_25"/>
      <StateTemplateVariable objectReference="ModelValue_26"/>
      <StateTemplateVariable objectReference="ModelValue_27"/>
      <StateTemplateVariable objectReference="ModelValue_28"/>
      <StateTemplateVariable objectReference="ModelValue_29"/>
      <StateTemplateVariable objectReference="ModelValue_30"/>
      <StateTemplateVariable objectReference="ModelValue_31"/>
      <StateTemplateVariable objectReference="ModelValue_32"/>
      <StateTemplateVariable objectReference="ModelValue_33"/>
      <StateTemplateVariable objectReference="ModelValue_34"/>
      <StateTemplateVariable objectReference="ModelValue_35"/>
      <StateTemplateVariable objectReference="ModelValue_36"/>
      <StateTemplateVariable objectReference="ModelValue_37"/>
      <StateTemplateVariable objectReference="ModelValue_38"/>
      <StateTemplateVariable objectReference="ModelValue_39"/>
      <StateTemplateVariable objectReference="ModelValue_40"/>
      <StateTemplateVariable objectReference="ModelValue_41"/>
      <StateTemplateVariable objectReference="ModelValue_42"/>
      <StateTemplateVariable objectReference="ModelValue_43"/>
      <StateTemplateVariable objectReference="ModelValue_44"/>
      <StateTemplateVariable objectReference="ModelValue_45"/>
      <StateTemplateVariable objectReference="ModelValue_46"/>
      <StateTemplateVariable objectReference="ModelValue_47"/>
      <StateTemplateVariable objectReference="ModelValue_48"/>
      <StateTemplateVariable objectReference="ModelValue_49"/>
      <StateTemplateVariable objectReference="ModelValue_50"/>
      <StateTemplateVariable objectReference="ModelValue_51"/>
      <StateTemplateVariable objectReference="ModelValue_52"/>
      <StateTemplateVariable objectReference="ModelValue_53"/>
      <StateTemplateVariable objectReference="ModelValue_54"/>
      <StateTemplateVariable objectReference="ModelValue_55"/>
      <StateTemplateVariable objectReference="ModelValue_56"/>
      <StateTemplateVariable objectReference="ModelValue_57"/>
      <StateTemplateVariable objectReference="ModelValue_58"/>
      <StateTemplateVariable objectReference="ModelValue_59"/>
    </StateTemplate>
    <InitialState type="initialState">
      0 9.0332112855000003e+26 0 4.5166056427500002e+26 0 0 0 0 0 0 6.0221408570000002e+23 0 0 1.8066422571000001e+27 0 0 9.6354253711999996e+26 0 0 6.0221408569999997e+25 6.0221408569999997e+25 0 1 0.02 50 1 0.01 1 0.01 20 1 2 1 0.5 1000 1 1 0.01 0.10000000000000001 0.001 0.10000000000000001 0.0080000000000000002 20 0.20000000000000001 1 0.01 50 1 0.00035 10 50 0.01 0.10000000000000001 0.01 1 10 50 0.050000000000000003 50 0.050000000000000003 50 0.050000000000000003 50 0.01 1 5 250 400 400 0.01 1 5 250 1 500 500 100 500 100 1 10 100 0 
    </InitialState>
  </Model>
  <ListOfTasks>
    <Task key="Task_26" name="Steady-State" type="steadyState" scheduled="false" updateModel="false">
      <Report reference="Report_17" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="JacobianRequested" type="bool" value="1"/>
        <Parameter name="StabilityAnalysisRequested" type="bool" value="1"/>
      </Problem>
      <Method name="Enhanced Newton" type="EnhancedNewton">
        <Parameter name="Resolution" type="unsignedFloat" value="1.0000000000000001e-09"/>
        <Parameter name="Derivation Factor" type="unsignedFloat" value="0.001"/>
        <Parameter name="Use Newton" type="bool" value="1"/>
        <Parameter name="Use Integration" type="bool" value="1"/>
        <Parameter name="Use Back Integration" type="bool" value="0"/>
        <Parameter name="Accept Negative Concentrations" type="bool" value="0"/>
        <Parameter name="Iteration Limit" type="unsignedInteger" value="50"/>
        <Parameter name="Maximum duration for forward integration" type="unsignedFloat" value="1000000000"/>
        <Parameter name="Maximum duration for backward integration" type="unsignedFloat" value="1000000"/>
      </Method>
    </Task>
    <Task key="Task_25" name="Time-Course" type="timeCourse" scheduled="false" updateModel="false">
      <Report reference="Report_0" target="copasi-ground-output" append="0" confirmOverwrite="0"/>
      <Problem>
        <Parameter name="AutomaticStepSize" type="bool" value="0"/>
        <Parameter name="StepNumber" type="unsignedInteger" value="10"/>
        <Parameter name="StepSize" type="float" value="10"/>
        <Parameter name="Duration" type="float" value="100"/>
        <Parameter name="TimeSeriesRequested" type="bool" value="1"/>
        <Parameter name="OutputStartTime" type="float" value="0"/>
        <Parameter name="Output Event" type="bool" value="0"/>
        <Parameter name="Start in Steady State" type="bool" value="0"/>
      </Problem>
      <Method name="Deterministic (LSODA)" type="Deterministic(LSODA)">
        <Parameter name="Integrate Reduced Model" type="bool" value="0"/>
        <Parameter name="Relative Tolerance" type="unsignedFloat" value="9.9999999999999995e-07"/>
        <Parameter name="Absolute Tolerance" type="unsignedFloat" value="9.9999999999999998e-13"/>
        <Parameter name="Max Internal Steps" type="unsignedInteger" value="10000"/>
        <Parameter name="Max Internal Step Size" type="unsignedFloat" value="0"/>
      </Method>
    </Task>
    <Task key="Task_24" name="Scan" type="scan" scheduled="false" updateModel="false">
      <Problem>
        <Parameter name="Subtask" type="unsignedInteger" value="1"/>
        <ParameterGroup name="ScanItems">
        </ParameterGroup>
        <Parameter name="Output in subtask" type="bool" value="1"/>
        <Parameter name="Adjust initial conditions" type="bool" value="0"/>
      </Problem>
      <Method name="Scan Framework" type="ScanFramework">
      </Method>
    </Task>
    <Task key="Task_23" name="Elementary Flux Modes" type="fluxMode" scheduled="false" updateModel="false">
      <Report reference="Report_16" target="" append="1" confirmOverwrite="1"/>
      <Problem>
      </Problem>
      <Method name="EFM Algorithm" type="EFMAlgorithm">
      </Method>
    </Task>
    <Task key="Task_22" name="Optimization" type="optimization" scheduled="false" updateModel="false">
      <Report reference="Report_15" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="Subtask" type="cn" value="CN=Root,Vector=TaskList[Steady-State]"/>
        <ParameterText name="ObjectiveExpression" type="expression">
          
        </ParameterText>
        <Parameter name="Maximize" type="bool" value="0"/>
        <Parameter name="Randomize Start Values" type="bool" value="0"/>
        <Parameter name="Calculate Statistics" type="bool" value="1"/>
        <ParameterGroup name="OptimizationItemList">
        </ParameterGroup>
        <ParameterGroup name="OptimizationConstraintList">
        </ParameterGroup>
      </Problem>
      <Method name="Random Search" type="RandomSearch">
        <Parameter name="Number of Iterations" type="unsignedInteger" value="100000"/>
        <Parameter name="Random Number Generator" type="unsignedInteger" value="1"/>
        <Parameter name="Seed" type="unsignedInteger" value="0"/>
      </Method>
    </Task>
    <Task key="Task_21" name="Parameter Estimation" type="parameterFitting" scheduled="false" updateModel="false">
      <Report reference="Report_14" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="Maximize" type="bool" value="0"/>
        <Parameter name="Randomize Start Values" type="bool" value="0"/>
        <Parameter name="Calculate Statistics" type="bool" value="1"/>
        <ParameterGroup name="OptimizationItemList">
        </ParameterGroup>
        <ParameterGroup name="OptimizationConstraintList">
        </ParameterGroup>
        <Parameter name="Steady-State" type="cn" value="CN=Root,Vector=TaskList[Steady-State]"/>
        <Parameter name="Time-Course" type="cn" value="CN=Root,Vector=TaskList[Time-Course]"/>
        <Parameter name="Create Parameter Sets" type="bool" value="0"/>
        <ParameterGroup name="Experiment Set">
        </ParameterGroup>
        <ParameterGroup name="Validation Set">
          <Parameter name="Weight" type="unsignedFloat" value="1"/>
          <Parameter name="Threshold" type="unsignedInteger" value="5"/>
        </ParameterGroup>
      </Problem>
      <Method name="Evolutionary Programming" type="EvolutionaryProgram">
        <Parameter name="Number of Generations" type="unsignedInteger" value="200"/>
        <Parameter name="Population Size" type="unsignedInteger" value="20"/>
        <Parameter name="Random Number Generator" type="unsignedInteger" value="1"/>
        <Parameter name="Seed" type="unsignedInteger" value="0"/>
      </Method>
    </Task>
    <Task key="Task_20" name="Metabolic Control Analysis" type="metabolicControlAnalysis" scheduled="false" updateModel="false">
      <Report reference="Report_13" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="Steady-State" type="key" value="Task_26"/>
      </Problem>
      <Method name="MCA Method (Reder)" type="MCAMethod(Reder)">
        <Parameter name="Modulation Factor" type="unsignedFloat" value="1.0000000000000001e-09"/>
        <Parameter name="Use Reder" type="bool" value="1"/>
        <Parameter name="Use Smallbone" type="bool" value="1"/>
      </Method>
    </Task>
    <Task key="Task_19" name="Lyapunov Exponents" type="lyapunovExponents" scheduled="false" updateModel="false">
      <Report reference="Report_12" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="ExponentNumber" type="unsignedInteger" value="3"/>
        <Parameter name="DivergenceRequested" type="bool" value="1"/>
        <Parameter name="TransientTime" type="float" value="0"/>
      </Problem>
      <Method name="Wolf Method" type="WolfMethod">
        <Parameter name="Orthonormalization Interval" type="unsignedFloat" value="1"/>
        <Parameter name="Overall time" type="unsignedFloat" value="1000"/>
        <Parameter name="Relative Tolerance" type="unsignedFloat" value="9.9999999999999995e-07"/>
        <Parameter name="Absolute Tolerance" type="unsignedFloat" value="9.9999999999999998e-13"/>
        <Parameter name="Max Internal Steps" type="unsignedInteger" value="10000"/>
      </Method>
    </Task>
    <Task key="Task_18" name="Time Scale Separation Analysis" type="timeScaleSeparationAnalysis" scheduled="false" updateModel="false">
      <Report reference="Report_11" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="StepNumber" type="unsignedInteger" value="100"/>
        <Parameter name="StepSize" type="float" value="0.01"/>
        <Parameter name="Duration" type="float" value="1"/>
        <Parameter name="TimeSeriesRequested" type="bool" value="1"/>
        <Parameter name="OutputStartTime" type="float" value="0"/>
      </Problem>
      <Method name="ILDM (LSODA,Deuflhard)" type="TimeScaleSeparation(ILDM,Deuflhard)">
        <Parameter name="Deuflhard Tolerance" type="unsignedFloat" value="0.0001"/>
      </Method>
    </Task>
    <Task key="Task_17" name="Sensitivities" type="sensitivities" scheduled="false" updateModel="false">
      <Report reference="Report_10" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="SubtaskType" type="unsignedInteger" value="1"/>
        <ParameterGroup name="TargetFunctions">
          <Parameter name="SingleObject" type="cn" value=""/>
          <Parameter name="ObjectListType" type="unsignedInteger" value="7"/>
        </ParameterGroup>
        <ParameterGroup name="ListOfVariables">
          <ParameterGroup name="Variables">
            <Parameter name="SingleObject" type="cn" value=""/>
            <Parameter name="ObjectListType" type="unsignedInteger" value="41"/>
          </ParameterGroup>
          <ParameterGroup name="Variables">
            <Parameter name="SingleObject" type="cn" value=""/>
            <Parameter name="ObjectListType" type="unsignedInteger" value="0"/>
          </ParameterGroup>
        </ParameterGroup>
      </Problem>
      <Method name="Sensitivities Method" type="SensitivitiesMethod">
        <Parameter name="Delta factor" type="unsignedFloat" value="0.001"/>
        <Parameter name="Delta minimum" type="unsignedFloat" value="9.9999999999999998e-13"/>
      </Method>
    </Task>
    <Task key="Task_16" name="Moieties" type="moieties" scheduled="false" updateModel="false">
      <Problem>
      </Problem>
      <Method name="Householder Reduction" type="Householder">
      </Method>
    </Task>
    <Task key="Task_15" name="Cross Section" type="crosssection" scheduled="false" updateModel="false">
      <Problem>
        <Parameter name="AutomaticStepSize" type="bool" value="0"/>
        <Parameter name="StepNumber" type="unsignedInteger" value="100"/>
        <Parameter name="StepSize" type="float" value="0.01"/>
        <Parameter name="Duration" type="float" value="1"/>
        <Parameter name="TimeSeriesRequested" type="bool" value="1"/>
        <Parameter name="OutputStartTime" type="float" value="0"/>
        <Parameter name="Output Event" type="bool" value="0"/>
        <Parameter name="Start in Steady State" type="bool" value="0"/>
        <Parameter name="LimitCrossings" type="bool" value="0"/>
        <Parameter name="NumCrossingsLimit" type="unsignedInteger" value="0"/>
        <Parameter name="LimitOutTime" type="bool" value="0"/>
        <Parameter name="LimitOutCrossings" type="bool" value="0"/>
        <Parameter name="PositiveDirection" type="bool" value="1"/>
        <Parameter name="NumOutCrossingsLimit" type="unsignedInteger" value="0"/>
        <Parameter name="LimitUntilConvergence" type="bool" value="0"/>
        <Parameter name="ConvergenceTolerance" type="float" value="9.9999999999999995e-07"/>
        <Parameter name="Threshold" type="float" value="0"/>
        <Parameter name="DelayOutputUntilConvergence" type="bool" value="0"/>
        <Parameter name="OutputConvergenceTolerance" type="float" value="9.9999999999999995e-07"/>
        <ParameterText name="TriggerExpression" type="expression">
          
        </ParameterText>
        <Parameter name="SingleVariable" type="cn" value=""/>
      </Problem>
      <Method name="Deterministic (LSODA)" type="Deterministic(LSODA)">
        <Parameter name="Integrate Reduced Model" type="bool" value="0"/>
        <Parameter name="Relative Tolerance" type="unsignedFloat" value="9.9999999999999995e-07"/>
        <Parameter name="Absolute Tolerance" type="unsignedFloat" value="9.9999999999999998e-13"/>
        <Parameter name="Max Internal Steps" type="unsignedInteger" value="10000"/>
        <Parameter name="Max Internal Step Size" type="unsignedFloat" value="0"/>
      </Method>
    </Task>
    <Task key="Task_27" name="Linear Noise Approximation" type="linearNoiseApproximation" scheduled="false" updateModel="false">
      <Report reference="Report_9" target="" append="1" confirmOverwrite="1"/>
      <Problem>
        <Parameter name="Steady-State" type="key" value="Task_26"/>
      </Problem>
      <Method name="Linear Noise Approximation" type="LinearNoiseApproximation">
      </Method>
    </Task>
  </ListOfTasks>
  <ListOfReports>
    <Report key="Report_17" name="Steady-State" taskType="steadyState" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Footer>
        <Object cn="CN=Root,Vector=TaskList[Steady-State]"/>
      </Footer>
    </Report>
    <Report key="Report_16" name="Elementary Flux Modes" taskType="fluxMode" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Footer>
        <Object cn="CN=Root,Vector=TaskList[Elementary Flux Modes],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_15" name="Optimization" taskType="optimization" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Optimization],Object=Description"/>
        <Object cn="String=\[Function Evaluations\]"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="String=\[Best Value\]"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="String=\[Best Parameters\]"/>
      </Header>
      <Body>
        <Object cn="CN=Root,Vector=TaskList[Optimization],Problem=Optimization,Reference=Function Evaluations"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="CN=Root,Vector=TaskList[Optimization],Problem=Optimization,Reference=Best Value"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="CN=Root,Vector=TaskList[Optimization],Problem=Optimization,Reference=Best Parameters"/>
      </Body>
      <Footer>
        <Object cn="String=&#x0a;"/>
        <Object cn="CN=Root,Vector=TaskList[Optimization],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_14" name="Parameter Estimation" taskType="parameterFitting" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Parameter Estimation],Object=Description"/>
        <Object cn="String=\[Function Evaluations\]"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="String=\[Best Value\]"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="String=\[Best Parameters\]"/>
      </Header>
      <Body>
        <Object cn="CN=Root,Vector=TaskList[Parameter Estimation],Problem=Parameter Estimation,Reference=Function Evaluations"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="CN=Root,Vector=TaskList[Parameter Estimation],Problem=Parameter Estimation,Reference=Best Value"/>
        <Object cn="Separator=&#x09;"/>
        <Object cn="CN=Root,Vector=TaskList[Parameter Estimation],Problem=Parameter Estimation,Reference=Best Parameters"/>
      </Body>
      <Footer>
        <Object cn="String=&#x0a;"/>
        <Object cn="CN=Root,Vector=TaskList[Parameter Estimation],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_13" name="Metabolic Control Analysis" taskType="metabolicControlAnalysis" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Metabolic Control Analysis],Object=Description"/>
      </Header>
      <Footer>
        <Object cn="String=&#x0a;"/>
        <Object cn="CN=Root,Vector=TaskList[Metabolic Control Analysis],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_12" name="Lyapunov Exponents" taskType="lyapunovExponents" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Lyapunov Exponents],Object=Description"/>
      </Header>
      <Footer>
        <Object cn="String=&#x0a;"/>
        <Object cn="CN=Root,Vector=TaskList[Lyapunov Exponents],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_11" name="Time Scale Separation Analysis" taskType="timeScaleSeparationAnalysis" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Time Scale Separation Analysis],Object=Description"/>
      </Header>
      <Footer>
        <Object cn="String=&#x0a;"/>
        <Object cn="CN=Root,Vector=TaskList[Time Scale Separation Analysis],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_10" name="Sensitivities" taskType="sensitivities" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Sensitivities],Object=Description"/>
      </Header>
      <Footer>
        <Object cn="String=&#x0a;"/>
        <Object cn="CN=Root,Vector=TaskList[Sensitivities],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_9" name="Linear Noise Approximation" taskType="linearNoiseApproximation" separator="&#x09;" precision="6">
      <Comment>
        Automatically generated report.
      </Comment>
      <Header>
        <Object cn="CN=Root,Vector=TaskList[Linear Noise Approximation],Object=Description"/>
      </Header>
      <Footer>
        <Object cn="String=&#x0a;"/>
        <Object cn="CN=Root,Vector=TaskList[Linear Noise Approximation],Object=Result"/>
      </Footer>
    </Report>
    <Report key="Report_0" name="Time Course" taskType="timeCourse" separator="&#x09;" precision="6">
      <Comment>
      </Comment>
      <Table printTitle="1">
        <Object cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Reference=Time"/>
        <Object cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment],Vector=Metabolites[ERK],Reference=Concentration"/>
        <Object cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment],Vector=Metabolites[MEK],Reference=Concentration"/>
        <Object cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment],Vector=Metabolites[pERK],Reference=Concentration"/>
        <Object cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment],Vector=Metabolites[pMEK],Reference=Concentration"/>
        <Object cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment],Vector=Metabolites[ppERK],Reference=Concentration"/>
        <Object cn="CN=Root,Model=Romano-Nguyen2014 -  MST2 and Raf1 Crosstalk,Vector=Compartments[default_compartment],Vector=Metabolites[ppMEK],Reference=Concentration"/>
      </Table>
    </Report>
  </ListOfReports>
  <GUI>
  </GUI>
  <SBMLReference file="Romano2014.xml">
    <SBMLMap SBMLid="AKTa" COPASIkey="ModelValue_19"/>
    <SBMLMap SBMLid="ERK" COPASIkey="Metabolite_26"/>
    <SBMLMap SBMLid="F1A" COPASIkey="Metabolite_17"/>
    <SBMLMap SBMLid="Fa" COPASIkey="ModelValue_10"/>
    <SBMLMap SBMLid="Ka" COPASIkey="ModelValue_11"/>
    <SBMLMap SBMLid="Kact" COPASIkey="ModelValue_20"/>
    <SBMLMap SBMLid="Ki" COPASIkey="ModelValue_56"/>
    <SBMLMap SBMLid="Kin" COPASIkey="ModelValue_6"/>
    <SBMLMap SBMLid="Km1" COPASIkey="ModelValue_2"/>
    <SBMLMap SBMLid="Km10" COPASIkey="ModelValue_27"/>
    <SBMLMap SBMLid="Km13" COPASIkey="ModelValue_33"/>
    <SBMLMap SBMLid="Km14a" COPASIkey="ModelValue_35"/>
    <SBMLMap SBMLid="Km14b" COPASIkey="ModelValue_37"/>
    <SBMLMap SBMLid="Km15" COPASIkey="ModelValue_39"/>
    <SBMLMap SBMLid="Km17" COPASIkey="ModelValue_44"/>
    <SBMLMap SBMLid="Km19" COPASIkey="ModelValue_45"/>
    <SBMLMap SBMLid="Km20" COPASIkey="ModelValue_51"/>
    <SBMLMap SBMLid="Km21" COPASIkey="ModelValue_54"/>
    <SBMLMap SBMLid="Km22" COPASIkey="ModelValue_52"/>
    <SBMLMap SBMLid="Km23" COPASIkey="ModelValue_55"/>
    <SBMLMap SBMLid="Km2a" COPASIkey="ModelValue_4"/>
    <SBMLMap SBMLid="Km2b" COPASIkey="ModelValue_7"/>
    <SBMLMap SBMLid="Km3" COPASIkey="ModelValue_9"/>
    <SBMLMap SBMLid="Km4" COPASIkey="ModelValue_13"/>
    <SBMLMap SBMLid="Km7" COPASIkey="ModelValue_21"/>
    <SBMLMap SBMLid="Km8" COPASIkey="ModelValue_24"/>
    <SBMLMap SBMLid="LATS1" COPASIkey="Metabolite_20"/>
    <SBMLMap SBMLid="LATS1a" COPASIkey="Metabolite_9"/>
    <SBMLMap SBMLid="MEK" COPASIkey="Metabolite_21"/>
    <SBMLMap SBMLid="MST2" COPASIkey="Metabolite_14"/>
    <SBMLMap SBMLid="MST2a" COPASIkey="Metabolite_16"/>
    <SBMLMap SBMLid="M_F1A" COPASIkey="Metabolite_19"/>
    <SBMLMap SBMLid="M_pRi" COPASIkey="Metabolite_15"/>
    <SBMLMap SBMLid="Ma_F1A" COPASIkey="Metabolite_18"/>
    <SBMLMap SBMLid="PP2A" COPASIkey="ModelValue_23"/>
    <SBMLMap SBMLid="Ra_Mk" COPASIkey="Metabolite_22"/>
    <SBMLMap SBMLid="Ra_pMk" COPASIkey="Metabolite_25"/>
    <SBMLMap SBMLid="Raf_1" COPASIkey="Metabolite_8"/>
    <SBMLMap SBMLid="Raf_1a" COPASIkey="Metabolite_10"/>
    <SBMLMap SBMLid="RasGTP" COPASIkey="ModelValue_1"/>
    <SBMLMap SBMLid="V13" COPASIkey="ModelValue_32"/>
    <SBMLMap SBMLid="V15" COPASIkey="ModelValue_38"/>
    <SBMLMap SBMLid="V17" COPASIkey="ModelValue_43"/>
    <SBMLMap SBMLid="V19" COPASIkey="ModelValue_49"/>
    <SBMLMap SBMLid="V21" COPASIkey="ModelValue_53"/>
    <SBMLMap SBMLid="V23" COPASIkey="ModelValue_58"/>
    <SBMLMap SBMLid="V3" COPASIkey="ModelValue_8"/>
    <SBMLMap SBMLid="V4" COPASIkey="ModelValue_12"/>
    <SBMLMap SBMLid="default_compartment" COPASIkey="Compartment_3"/>
    <SBMLMap SBMLid="k1" COPASIkey="ModelValue_0"/>
    <SBMLMap SBMLid="k10" COPASIkey="ModelValue_26"/>
    <SBMLMap SBMLid="k11f" COPASIkey="ModelValue_28"/>
    <SBMLMap SBMLid="k11r" COPASIkey="ModelValue_29"/>
    <SBMLMap SBMLid="k12f" COPASIkey="ModelValue_30"/>
    <SBMLMap SBMLid="k12r" COPASIkey="ModelValue_31"/>
    <SBMLMap SBMLid="k14a" COPASIkey="ModelValue_34"/>
    <SBMLMap SBMLid="k14b" COPASIkey="ModelValue_36"/>
    <SBMLMap SBMLid="k16af" COPASIkey="ModelValue_40"/>
    <SBMLMap SBMLid="k16ar" COPASIkey="ModelValue_41"/>
    <SBMLMap SBMLid="k16b" COPASIkey="ModelValue_42"/>
    <SBMLMap SBMLid="k18af" COPASIkey="ModelValue_46"/>
    <SBMLMap SBMLid="k18ar" COPASIkey="ModelValue_47"/>
    <SBMLMap SBMLid="k18b" COPASIkey="ModelValue_48"/>
    <SBMLMap SBMLid="k20" COPASIkey="ModelValue_50"/>
    <SBMLMap SBMLid="k22" COPASIkey="ModelValue_57"/>
    <SBMLMap SBMLid="k2a" COPASIkey="ModelValue_3"/>
    <SBMLMap SBMLid="k2b" COPASIkey="ModelValue_5"/>
    <SBMLMap SBMLid="k5f" COPASIkey="ModelValue_14"/>
    <SBMLMap SBMLid="k5r" COPASIkey="ModelValue_15"/>
    <SBMLMap SBMLid="k6f" COPASIkey="ModelValue_16"/>
    <SBMLMap SBMLid="k6r" COPASIkey="ModelValue_17"/>
    <SBMLMap SBMLid="k7" COPASIkey="ModelValue_18"/>
    <SBMLMap SBMLid="k8" COPASIkey="ModelValue_22"/>
    <SBMLMap SBMLid="k9" COPASIkey="ModelValue_25"/>
    <SBMLMap SBMLid="pERK" COPASIkey="Metabolite_27"/>
    <SBMLMap SBMLid="pMEK" COPASIkey="Metabolite_23"/>
    <SBMLMap SBMLid="pMST2" COPASIkey="ModelValue_59"/>
    <SBMLMap SBMLid="pMST2i" COPASIkey="Metabolite_12"/>
    <SBMLMap SBMLid="pMi_pRi" COPASIkey="Metabolite_13"/>
    <SBMLMap SBMLid="pRaf_1i" COPASIkey="Metabolite_7"/>
    <SBMLMap SBMLid="ppERK" COPASIkey="Metabolite_11"/>
    <SBMLMap SBMLid="ppMEK" COPASIkey="Metabolite_24"/>
    <SBMLMap SBMLid="v1" COPASIkey="Reaction_10"/>
    <SBMLMap SBMLid="v10" COPASIkey="Reaction_21"/>
    <SBMLMap SBMLid="v11a" COPASIkey="Reaction_22"/>
    <SBMLMap SBMLid="v11b" COPASIkey="Reaction_23"/>
    <SBMLMap SBMLid="v12a" COPASIkey="Reaction_24"/>
    <SBMLMap SBMLid="v12b" COPASIkey="Reaction_25"/>
    <SBMLMap SBMLid="v13" COPASIkey="Reaction_26"/>
    <SBMLMap SBMLid="v14" COPASIkey="Reaction_27"/>
    <SBMLMap SBMLid="v15" COPASIkey="Reaction_28"/>
    <SBMLMap SBMLid="v16a" COPASIkey="Reaction_29"/>
    <SBMLMap SBMLid="v16aa" COPASIkey="Reaction_30"/>
    <SBMLMap SBMLid="v16b" COPASIkey="Reaction_31"/>
    <SBMLMap SBMLid="v17" COPASIkey="Reaction_32"/>
    <SBMLMap SBMLid="v18a" COPASIkey="Reaction_33"/>
    <SBMLMap SBMLid="v18aa" COPASIkey="Reaction_34"/>
    <SBMLMap SBMLid="v18b" COPASIkey="Reaction_35"/>
    <SBMLMap SBMLid="v19" COPASIkey="Reaction_36"/>
    <SBMLMap SBMLid="v2" COPASIkey="Reaction_11"/>
    <SBMLMap SBMLid="v20" COPASIkey="Reaction_37"/>
    <SBMLMap SBMLid="v21" COPASIkey="Reaction_38"/>
    <SBMLMap SBMLid="v22" COPASIkey="Reaction_39"/>
    <SBMLMap SBMLid="v23" COPASIkey="Reaction_40"/>
    <SBMLMap SBMLid="v3" COPASIkey="Reaction_12"/>
    <SBMLMap SBMLid="v4" COPASIkey="Reaction_13"/>
    <SBMLMap SBMLid="v5a" COPASIkey="Reaction_14"/>
    <SBMLMap SBMLid="v5b" COPASIkey="Reaction_15"/>
    <SBMLMap SBMLid="v6a" COPASIkey="Reaction_16"/>
    <SBMLMap SBMLid="v6b" COPASIkey="Reaction_17"/>
    <SBMLMap SBMLid="v7" COPASIkey="Reaction_18"/>
    <SBMLMap SBMLid="v8" COPASIkey="Reaction_19"/>
    <SBMLMap SBMLid="v9" COPASIkey="Reaction_20"/>
  </SBMLReference>
  <ListOfUnitDefinitions>
    <UnitDefinition key="Unit_0" name="meter" symbol="m">
      <Expression>
        m
      </Expression>
    </UnitDefinition>
    <UnitDefinition key="Unit_2" name="second" symbol="s">
      <Expression>
        s
      </Expression>
    </UnitDefinition>
    <UnitDefinition key="Unit_6" name="Avogadro" symbol="Avogadro">
      <Expression>
        Avogadro
      </Expression>
    </UnitDefinition>
    <UnitDefinition key="Unit_8" name="item" symbol="#">
      <Expression>
        #
      </Expression>
    </UnitDefinition>
  </ListOfUnitDefinitions>
</COPASI>
