import React, { useState, useEffect, useRef } from 'react';
import styled, { keyframes } from 'styled-components';
import axios from 'axios';

const Container = styled.div`
  padding: 20px;
  position: relative;
  background-color: white;
  min-height: 100vh;
  background-image: url('/ai-data-festival-bg.jpg');
  background-size: cover;
  background-position: center;
`;

const ContentWrapper = styled.div`
  max-width: 800px;
  margin: 0 auto;
  background-color: rgba(255, 255, 255, 0.95);
  padding: 10%;
  border-radius: 10px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
`;

const Title = styled.h1`
  color: #003366;
  text-align: center;
  margin-bottom: 30px;
`;

const InputWrapper = styled.div`
  position: relative;
  margin-bottom: 20px;
`;

const Input = styled.textarea`
  width: 100%;
  margin-right: 200%;
  min-height: 350px;
  padding: 15px;
  font-size: 16px;
  border: none;
  border-radius: 24px;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
  resize: vertical;
  outline: none;
  transition: box-shadow 0.3s ease;

  &:focus {
    box-shadow: 0 3px 8px rgba(0, 0, 0, 0.15);
  }
`;

const ButtonContainer = styled.div`
  display: flex;
  justify-content: center;
  margin-top: 20px;
`;

const Button = styled.button`
  background-color: #FF6600;
  color: white;
  padding: 12px 30px;
  border: none;
  border-radius: 24px;
  cursor: pointer;
  font-size: 18px;
  transition: background-color 0.3s, transform 0.1s;

  &:hover {
    background-color: #FF8533;
  }

  &:active {
    transform: scale(0.98);
  }
`;

const ResultContainer = styled.div`
  margin-top: 20px;
  opacity: ${props => props.show ? 1 : 0};
  transition: opacity 0.5s ease-in-out;
  max-height: 400px;
  overflow-y: auto;
  background-color: white;
  padding: 20px;
  border-radius: 4px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
`;

const Overlay = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 51, 102, 0.7);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
`;

const pulse = keyframes`
  0% { opacity: 0.5; }
  50% { opacity: 1; }
  100% { opacity: 0.5; }
`;

const LoadingText = styled.div`
  color: white;
  font-size: 24px;
  animation: ${pulse} 1.5s infinite ease-in-out;
`;

const fadeIn = keyframes`
  from { opacity: 0; }
  to { opacity: 1; }
`;

const KeywordItem = styled.div`
  animation: ${fadeIn} 0.5s ease-in-out;
  margin-bottom: 10px;
  padding: 10px;
  background-color: ${props => props.sentiment === '긍정' ? '#e6f7ff' : '#ffe6e6'};
  border-radius: 4px;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const Keyword = styled.span`
  font-weight: bold;
  color: #003366;
  flex: 1;
`;

const Sentiment = styled.span`
  color: ${props => props.sentiment === '긍정' ? '#52c41a' : '#f5222d'};
  flex: 0 0 60px;
  text-align: center;
`;

const Ratio = styled.span`
  color: #1890ff;
  flex: 0 0 60px;
  text-align: right;
`;

function NewsAnalysis() {
  const [article, setArticle] = useState('');
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showResult, setShowResult] = useState(false);
  const resultRef = useRef(null);

  useEffect(() => {
    if (result && resultRef.current) {
      resultRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [result]);

  const handleAnalysis = async () => {
    setIsLoading(true);
    setShowResult(false);
    try {
      const response = await axios.post('http://localhost:8100/generate_keywords', { article });
      setResult(response.data.selected_keywords);
      setShowResult(true);
    } catch (error) {
      console.error('Error:', error);
      alert('분석 중 오류가 발생했습니다.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Container>
      <ContentWrapper>
        <Title>AI·Data Festival 키워드 분석</Title>
        <InputWrapper>
          <Input 
            value={article} 
            onChange={(e) => setArticle(e.target.value)} 
            placeholder="뉴스 기사를 입력하세요..."
          />
        </InputWrapper>
        <ButtonContainer>
          <Button onClick={handleAnalysis}>분석하기</Button>
        </ButtonContainer>
        {isLoading && (
          <Overlay>
            <LoadingText>분석중...</LoadingText>
          </Overlay>
        )}
        <ResultContainer show={showResult} ref={resultRef}>
          {result && result.map((item, index) => (
            <KeywordItem key={index} sentiment={item.sentiment}>
              <Keyword>{item.keyword}</Keyword>
              <Sentiment sentiment={item.sentiment}>{item.sentiment}</Sentiment>
              <Ratio>{item.ratio}%</Ratio>
            </KeywordItem>
          ))}
        </ResultContainer>
      </ContentWrapper>
    </Container>
  );
}

export default NewsAnalysis;