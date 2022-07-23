var trArr = document.querySelector("#printZone > table:nth-child(2) > tbody tr")
//                                                  nth-child : css 언어 (index와 달리 1번부터 시작함)
//                                   부등호 사용 : 바로 아래 것만 해당, 부등호 안사용 : 아래있는 모든것 뽑겠다

// [약국명 * 주소 * 전화번호 * 운영시간] 긁고 나서 담을 빈 리스트 생성
var returnData = []

for (var i=0; i < trArr.length; i++) {
    
    var currentTr = trArr[i]

    // 약국명
    var name = currentTr.querySelector('td')?.innerText.replace('\n', '').replace('\t', '')

    // 주소
    var address = currentTr.querySelector('td')[2]?.innerText.replaceAll('\n', '').replaceAll('\t', '')

    // 전화번호
    var tel = currentTr.querySelector('td')[3]?.innerText.replaceAll('\n', '').replaceAll('\t', '')

    // 운영시간
    var open = currentTr.querySelector('td')[4]?.innerText.replaceAll('\n', '').replaceAll('\t', '')

    var jsonData = { name, address, tel, open }

    // var jsonData = {
    //     'name' : name,
    //     'address' : address,
    //     'tel' : tel,
    //     open
    // }
    // 이런식으로 써도 됨 -> 'open' : open 또는 open 만 써도 가능

    if ( jsonData.address != undefined) { // 만약 jsonData의 address가 undefined가 아니면
        returnData.push(jsonData) // returnData 빈 리스트 만든것에 넣어라(push)
    }
}
