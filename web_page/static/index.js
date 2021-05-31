
//헤더 불러오기.
$(document).ready(function(){    
    //버튼 클릭시 처리
    $("#btn").click(function(){
        var text = $("#inp").val();
	consol.log("hihi")
        move();
        search(text);
    });
    //글에서 엔터시 처리
    $("#inp").keydown(function(key){
        if(key.keyCode == 13) {
            var text = $("#inp").val();
	    consol.log("hihi")
	    move();
            search(text);
        }
    });
    
    //페이지 이동
    function move(){
        var offset = $('#accordion').offset(); //선택한 태그의 위치를 반환
        console.log(offset);
        offset.top = 280;
        console.log(offset);
        //animate()메서드를 이용해서 선택한 태그의 스크롤 위치를 지정해서 0.4초 동안 부드럽게 해당 위치로 이동함 
        $('html').animate({scrollTop : offset.top}, 400);
        loading();
    }

    //로딩 창
    function loading(){

    }

    //비동기 요청
    function search(text){
        $.ajax({
            url: '/search',
            dataType : 'json', //서버가 보내주는 데이터 타입
            data: {//서버로 보낼 데이터(파라메터)
                word: text,
                num: '123'
            },
            success:function(result){//3.결과오면 화면에 반영
                var count = 1
                var text = "";
                $(result).each(function(index, word){
                    word.content = word.content.replaceAll(word.highlight,`<mark>`+word.highlight+`</mark>`);
                    if(count == 1){
                        text += `
                        <div class="card">
                        <a class="card-link" data-toggle="collapse" href="#collapse${count}">
                            <div class="card-header">
                                ${word.title}
                                <span style="float:right"><i class="fas fa-angle-double-down"></i></span>
                            </div>
                        </a>
                        <div id="collapse${count++}" class="collapse show" data-parent="#accordion">
                          <div class="card-body">
                            ${word.content}
                          </div>
                        </div>
                      </div>
                    `
                    } else {
                        text += `
                        <div class="card">
                            <a class="collapsed card-link" data-toggle="collapse" href="#collapse${count}">
                                <div class="card-header">
                                ${word.title}
                                <span style="float:right"><i class="fas fa-angle-double-down"></i></span>
                                </div>
                            </a>
                            <div id="collapse${count++}" class="collapse" data-parent="#accordion">
                                <div class="card-body">
                                    ${word.content}
                                    
                                </div>
                            </div>
                        </div>
                    `
                    }
                });
                $("#accordion").html(text);
            }
        });
    }
});
